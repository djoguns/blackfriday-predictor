# Black Friday Predictor
A tensorflow app that predicts purchases on black friday based on historical customer data.

# Install
```
git clone https://github.com/mleila/blackfriday-predictor
pip install -e blackfriday-predictor
```


# Usage
There are two binaries that you get when you install this package, train and predict. The first allows you to build and train your own deep neural network models and train them on the blackfriday dataset. The latter allows you to make predictions on test data based on the models you trained.

Here are some examples on how to use `train` and `predict`
## train
Specify the model's name
```bash
train --model-name my_model
```
Specify columns to represent as indicator columns
```bash
train --indicators "Age" "Gender"
```
Specify columns to represent as embedding columns
```bash
train --embeddings {"Product_ID":100, "User_ID":100}
```
where the key is the column name and value is the number of embedding dimensions.

Create new features by crossing existing features
```bash
train --crossings '["Age", "Gender", "ind", 1]' '["Age", "Occupation", "emb", 1, 5]'
```
Specify estimator type. Currently only DNN_REG is supported, WIDE_DEEP coming soon.
```bash
train --estimator-type DNN_REG
```
Specify the number of layers and units per each layer
```bash
train --hidden-units 100 200 400
```
Specify the number of training steps
```bash
train --training-steps 15000
```

## predict

Predict a single example
```bash
predict --records '{"User_ID":"1000004",
					"Product_ID":"P00128942",
					"Gender":"M",
					"Age":"46-50",
					"Occupation":7,
					"City_Category":"B",
					"Stay_In_Current_City_Years":"2",
					"Marital_Status":1,
					"Product_Category_1":1,
					"Product_Category_2":11,
					"Product_Category_3":0}'
```
Predict several example
```basg
predict --records '{"User_ID":"1000004, ..."}' '{"User_ID":"1000005, ..."}'
```

predict using a csv file
```bash
predict --fpath 'test.csv'
```

write predictions to a file
```bash
predict --outfile 'test.csv'
```

# Demo
```bash
mkdir my-project
cd my-project

train --model-name awesome-model \
--embeddings '{"Product_ID":100, "User_ID":100}' \
--crossings '["Age", "Gender", "ind", 1]' \
--hidden-units 500 400 300 200 \
--training-steps 15000

predict predict --model-path savedmodels/awesome-model \
--fpath test.csv \
--outfile mypredcitions.txt
```
