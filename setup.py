from setuptools import setup, find_packages

setup(
    name='longtrainer',
    version='0.1.0',
    packages=find_packages(),
    description='A production-ready alternative to Embedchain',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Endevsols',
    author_email='technology@endevsols.com',
    url='https://github.com/ENDEVSOLS/Long-Trainer',
    install_requires=open('requirements.txt').read().splitlines(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)