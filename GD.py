

def GD(x,y):
    m = 0
    b = 0
    n = 1000
    learn = 0.001
    k = len(x)
    for i in range(n):
        y_predicted = m*x + b
        error =     (1/k)    *      sum((y-y_predicted)**2)
        error_dm =  (-2/k)   *      sum(x*(y-y_predicted))
        error_db =  (-2/k)   *      sum(y-y_predicted)
        m= m - learn*error_dm
        b= b - learn*error_db
        return error , m ,b
