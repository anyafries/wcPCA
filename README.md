# to do

* write the readme
* properly reference stablePCA and multi-criteria... (in particular to clone multi-criteria...)


```bash
python -m venv venv_wcPCA
source venv_wcPCA/bin/activate

# navigate to where you want minPCA
git clone https://github.com/anyafries/minPCA.git
cd minPCA
pip install . --extra-index-url https://download.pytorch.org/whl/cpu

# navigate back to wcPCA
python -m pip install matplotlib
python -m pip install pandas
python -m pip install seaborn 
python -m pip install cvxpy 
python -m pip install fancyimpute 
```