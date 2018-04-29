# bookrater
Predicts user's book ratings based on a collaborative filtering model.
Part of [BookWeb](https://github.com/danieln96/BookWeb_SIAG)

## Installation
Use conda or pip to install. With conda it will look like this:
```
conda create -f environment.yml
```

## Exploration
Start the app with
```
FLASK_APP=bookrater.py flask run
```
Then go to the `localhost:5000/graphql`, you will see the GraphiQL interface.
You can make a query - GraphiQL gives you autocompletion and shows what types arguments should have.
Output will show on the right.
