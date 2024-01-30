from setuptools import setup

if __name__ == "__main__":
    setup(extras_require={'gpu':  ['faiss-gpu'], 'cpu':  ['faiss-cpu']})
