from setuptools import setup

setup(
    name='Nuc3DMap',
    version='1.1',
    py_modules=['Nuc3DMap'],
    install_requires=[
        'Click','scipy','numpy','matplotlib','pandas','seaborn','statsmodels','cooler','numba','psutil','pyranges','tqdm','deeptools','bwa','pairtools','pybedtools'
    ],
    entry_points='''
    [console_scripts]
    Nuc3DMap=Nuc3DMap:cli
    '''
)
