## Image classification Web API Sample (chainer)

### What is this?
This is a sample of Image classification web API.
In this sample, we predict object based on Cifar100 trained model.

You can replace cifar100 model with your trained model and modify some codes and test your model easily.


#### Implemented API

- GET /predict.json	 
	- required params
		- url or data (base64 encoded)
	- optional params
		- top: number of candidates (default = 3)
- POST /predict.json
	- required params
		- file
	- optional params
		- top: number of candidates (default = 3)	
- GET /labels.json 

#### API Test Pages

- /upload
- /upload2

#### Test on local 

```
$ python main.py

```

#### Deploy to heroku

```
$ git clone git@github.com:tanakatsu/chainer-classification-api-sample.git
$ heroku create
$ git push heroku master
$ heroku open
```

