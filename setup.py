from setuptools import setup, find_packages


if __name__ == '__main__':
    setup(
        name='console-builder',
        version='0.1.0',
        author='Khanh T.',
        author_email='khanhtd@gmail.com',
        description='Conveniently implementing a console program.',
        python_requires='>=3.10',
        include_package_data=True,
        packages=find_packages(),
        zip_safe=False,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.10',
        ]
    )
