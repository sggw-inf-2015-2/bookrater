# bookrater
Predicts user's book ratings based on a collaborative filtering model.
Part of [BookWeb](https://github.com/danieln96/BookWeb_SIAG)

## Installation
The recommended way is to use pipenv:
```
pipenv install --dev
```
There will be an error during locking, that's tolerable.

On Debian-based systems it's likely that you will also have to install Tkinter for
Python:
```
# apt install python3-tk
```

## Exploration
Start the app with
```
FLASK_APP=bookrater.py flask run
```
Then go to the `localhost:5000/graphql`, you will see the GraphiQL interface.
You can make a query - GraphiQL gives you autocompletion and shows what types arguments should have.
It also provides introspection into available operations and their arguments
with a button in the top right.

### Queries
You can query the model to get predicted ratings for pairs of the form `(<user id>, <book id>)`.
The server will return a list of predicted ratings.
Queries have the following form:
```
{
    predictedRatings(users: <list of user ids>, books: <list of book ids>)
}
```

### Mutations
You can retrain the model on new data.
```
mutation {
    retrain(users: <user ids>, books: <book ids>, ratings: <ratings>)
}
```