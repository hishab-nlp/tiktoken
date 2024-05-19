pip uninstall tiktoken -y
rm -rf build tiktoken.egg-info
pip install .
echo "tiktoken reinstall completed"
