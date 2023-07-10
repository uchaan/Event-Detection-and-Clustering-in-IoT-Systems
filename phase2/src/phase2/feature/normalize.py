from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def standard_scaler(X):
    scaler = StandardScaler()
    normalized_X = scaler.fit_transform(X)

    return normalized_X


def min_max_scaler(X):
    scaler = MinMaxScaler()
    normalized_X = scaler.fit_transform(X)

    return normalized_X


def robust_scaler(X):
    scaler = RobustScaler()
    normalized_X = scaler.fit_transform(X)

    return normalized_X
