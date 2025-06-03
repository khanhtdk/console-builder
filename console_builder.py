import os
import sys
from abc import ABC, abstractmethod
from collections import namedtuple
from copy import deepcopy
from functools import wraps
from typing import Optional, Any, Dict, List, Type, Union
from argparse import ArgumentParser, ArgumentTypeError, Action, Namespace, _HelpAction, _SubParsersAction, SUPPRESS


HandledError = namedtuple(
    'HandledError',
    'error_type exit_code printable',
    defaults=(None, False)
)


def _daemonize(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        self = args[0]
        assert isinstance(self, ConsoleBuilder)
        if not self.run_as_daemon:
            return f(*args, **kwargs)
        if not self.pid_file:
            raise TypeError('Running as daemon requires a valid path to store PID file')
        pid = os.fork()
        if pid == 0:
            sys.exit(f(*args, **kwargs))
        return 0
    return wrapped


class ConsoleBuilder(ABC):
    # Program info
    program: str | None = None
    description: str | None = None
    usage_info: str | None = None
    suppress_help: bool = False

    # Console layout definition
    console_layout: Dict[str, Any] | None = None

    # List of errors to be handled by the console
    handled_errors: List[HandledError] = []

    # More actions to be added to the parser just like what already existed,
    # for example, `store_true`.
    custom_actions: Dict[str, Type[Action] | str] = {}

    # A file path for storing PID when running as daemon.
    pid_file = None

    # Whether to run the console as a daemon. A valid `pid_file` is required if
    # `run_as_daemon` is `True`.
    run_as_daemon = False

    # Constants
    DEFAULT_RETVAL_FOR_HANDLED_ERRORS = 100

    def __new__(cls, *args, **kwargs):
        for attr in dir(cls):
            if not attr.startswith('type_'):
                continue
            method = getattr(cls, attr)
            if not callable(method):
                continue
            setattr(cls, attr, cls.typecheck(method))
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        """
        :param args:     An array of arguments.
        :param kwargs:   Keyword arguments for post-init stage.
        """
        # Build parser
        self._parsers: Dict[str, ArgumentParser] = {}
        self.parser: ArgumentParser = self.build_parser()

        # Parse arguments
        self.args: Namespace = self.parser.parse_args(args or None)

        if kwargs.pop('test_mode', False) is True:
            print(self.args)
        else:
            self.__post_init__(**kwargs)

    def __post_init__(self, **kwargs):
        """Post-init tasks"""

    def _write_pid(self):
        if self.pid_file:
            if os.path.isfile(self.pid_file):
                raise FileExistsError(self.pid_file)
            with open(self.pid_file, 'w') as fp:
                fp.write(str(os.getpid()))

    def _remove_pid(self):
        if self.pid_file and os.path.isfile(self.pid_file):
            os.remove(self.pid_file)

    def _daemon_running(self) -> bool:
        if self.pid_file and os.path.isfile(self.pid_file):
            with open(self.pid_file) as fp:
                pid = int(fp.read().strip())
            return os.path.exists(f'/proc/{pid}')
        return False

    @abstractmethod
    def main(self):
        """Main execution method"""

    @_daemonize
    def run(self):
        if self._daemon_running():
            print('Process is running already', file=sys.stderr)
            return self.DEFAULT_RETVAL_FOR_HANDLED_ERRORS
        try:
            self._write_pid()
            self.main()
            return 0
        except Exception as e:
            for (error_type, exit_code, printable) in self.handled_errors:
                if isinstance(e, error_type):
                    if printable:
                        print(e, file=sys.stderr)
                    return exit_code or self.DEFAULT_RETVAL_FOR_HANDLED_ERRORS
            raise
        finally:
            self._remove_pid()

    @staticmethod
    def typecheck(func):
        """Makes the wrapped function a type-checker"""
        @wraps(func)
        def wrapped(value):
            try:
                ret = func(value)
                if isinstance(ret, bool):
                    assert ret is True
                    return value
                return ret
            except:
                name = func.__name__.replace('_', ' ').strip()
                if name.startswith('type'):
                    name = name[4:].strip()
                if ' ' in name:
                    name = repr(name)
                raise ArgumentTypeError('invalid %s: %r' % (name, value))
        return wrapped

    @classmethod
    def _parser_factory(cls, key, definition=None, parent=None):
        """
        Builds an individual sub-parser
        """
        # Custom settings
        customs = {
            'positionals_title': None,
            'optionals_title': None,
            'make_command': False,
            'suppress_help': cls.suppress_help
        }

        # Read layout's settings
        settings = {}
        if definition and '.' in definition:
            settings = definition.pop('.')
            for setting in customs:
                if setting in settings:
                    customs[setting] = settings.pop(setting)

        # Unpack key
        if key == '.':
            assert parent is None
            kind, key = key, cls.program
        else:
            assert parent is not None
            kind, key = key.split(':', maxsplit=1)

        # Root parser
        if kind == '.':
            if 'prog' not in settings:
                settings['prog'] = cls.program
            if 'usage' not in settings:
                settings['usage'] = cls.usage_info
            if 'description' in settings:
                settings['description'] = cls.description
            parser = ArgumentParser(**settings)

        # Parsers' container
        elif kind == 'sub':
            settings['dest'] = key
            parser = parent.add_subparsers(**settings)

        # Sub parser
        elif kind == 'parser':
            settings['name'] = key
            if customs['make_command'] and cls.program:
                if isinstance(parent, ArgumentParser):
                    settings['prog'] = parent.prog + key
                elif isinstance(parent, _SubParsersAction):
                    settings['prog'] = parent._prog_prefix + key
            parser = parent.add_parser(**settings)

        # Group parser
        elif kind == 'group':
            if 'title' not in settings:
                settings['title'] = key
            parser = parent.add_argument_group(**settings)

        # Not supported kind
        else:
            raise NotImplementedError(kind)

        # Customize parser
        if isinstance(parser, ArgumentParser):
            # Suppress help argument if not being shown
            if customs['suppress_help']:
                for action in parser._actions:
                    if isinstance(action, _HelpAction):
                        action.help = SUPPRESS
                        break
            # Set custom title for optionals section
            if customs['optionals_title']:
                parser._optionals.title = customs['optionals_title']
            # Set custom title for positionals section
            if customs['positionals_title']:
                parser._positionals.title = customs['positionals_title']
            # Register custom actions
            if cls.custom_actions:
                for name, klass in cls.custom_actions.items():
                    if isinstance(klass, str):
                        klass = getattr(cls, klass)
                    assert issubclass(klass, Action)
                    parser.register('action', name, klass)

        # Return parser
        return parser

    @classmethod
    def _args_factory(cls, key, definition, parser):
        """
        Builds an individual argument for parser
        """
        # Read custom definition keys
        short = definition.pop('short', None)
        aliases = definition.pop('aliases', [])
        positional = definition.pop('position', False)
        type_ = definition.get('type')

        # Process argument type
        if type_ is not None and isinstance(type_, str):
            if hasattr(cls, type_):
                definition['type'] = getattr(cls, type_)
            elif hasattr(cls, 'type_' + type_):
                definition['type'] = getattr(cls, 'type_' + type_)

        # Parse positional argument
        if positional is True:
            args = (key,)
            if short:
                raise TypeError("got an unexpected keyword argument 'short'")
            if aliases:
                raise TypeError("got an unexpected keyword argument 'aliases'")

        # Parse parameter options
        else:
            args = ('--' + key,)
            if short is not None:
                args = ('-' + short,) + args
            if aliases:
                args += tuple('--' + i for i in aliases if i != key)

        # Build argument
        parser.add_argument(*args, **definition)

    @classmethod
    def build_layout(cls) -> Dict[str, Any]:
        return getattr(cls, 'console_layout', None) or {}

    def error(self, message, prog=None):
        """Raises error on a specific parser or program"""
        prog = prog or '.'
        main = self._parsers['.'].prog
        prog = ''.join(prog.strip().split())
        if prog.startswith(main):
            prog = prog[len(main):].strip()
        parser = self._parsers[prog]
        parser.error(str(message))

    def build_parser(self, layout=None, parser=None):
        """
        Build console's parser based on predefined layout
        """
        # Clone the console's layout
        layout = deepcopy(layout or self.build_layout())

        # Build main parser at first call
        if parser is None:
            parser = self._parser_factory('.', layout)
            self._parsers['.'] = parser

        # Parse layout and build console
        for key, value in layout.items():
            if ':' in key:
                sub = self._parser_factory(key, value, parser)
                if isinstance(sub, ArgumentParser):
                    main = self._parsers['.'].prog
                    prog = ''.join(sub.prog[len(main):].strip().split())
                    self._parsers[prog] = sub
                if value:
                    self.build_parser(value, sub)
            elif value:
                self._args_factory(key, value, parser)

        # Return final built parser
        return parser
