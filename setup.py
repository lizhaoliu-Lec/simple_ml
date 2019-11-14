from setuptools import setup, find_packages

setup(
    name='simple_machine_learning',
    version='0.1.4',
    description='simple: simple machine learning framework.',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
    ],
    author='lizhaoliu',
    url='https://github.com/lizhaoliu-Lec/simple_ml',
    author_email='2524463910@qq.com',
    license='MIT',
    packages=['simple_ml/ensemble', 'simple_ml/linear', 'simple_ml/nn', 'simple_ml/preprocessing', 'simple_ml/utils'],
    include_package_data=False,
    install_requires=[
        'matplotlib>=3.1.0',
        'numpy>=1.16.4',
        'scipy>=1.2.1',
        'tqdm>=4.32.1',
        'pandas>=0.24.2',
        'setuptools>=41.0.1',
        'Pillow>=6.2.1',
        'scikit_learn>=0.21.3',
    ],
    zip_safe=True,
)
