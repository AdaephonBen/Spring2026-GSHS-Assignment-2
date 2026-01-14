from sklearn.linear_model import LogisticRegression

def logistic_regression(X_train, y_train, X_test):
    '''
    Implements the logistic regression algorithm.
    '''
    # Your code here
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred
