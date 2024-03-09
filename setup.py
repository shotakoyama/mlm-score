import setuptools

setuptools.setup(
        name = 'mlm-score',
        version = '0.1.0',
        packages = setuptools.find_packages(),
        entry_points = {
            'console_scripts':[
                    'mlm-score = mlmscore.main:main',
                ]})

