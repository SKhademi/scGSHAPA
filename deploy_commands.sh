# update buikd
# pip install --upgrade build
python -m build
# Upload the package to pypi repository
#twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
twine upload --repository pypi dist/*
