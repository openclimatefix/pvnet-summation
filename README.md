# OCF Template Repository
This project is used for training a model to sum the GSP predictions of [PVNet](https://github.com/openclimatefix/PVNet) into a national estimate.

## Setup
```bash
git clone https://github.com/openclimatefix/PVNet_summation
cd PVNet_summation
pip install -r requirements.txt
pip install git+https://github.com/SheffieldSolar/PV_Live-API
```

Do the following to customize the repo to the project:

- Add PyPi access token to release to PyPi
- Update name of folder in the test workflow
