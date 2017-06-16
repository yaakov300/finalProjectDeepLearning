training = None;

if(training == None):
    training = 0;

def update_training(num):
    global training
    training = num

def incress_training():
    if (training == None):
        global training
        training = 0
    else:
        global training
        training +=1
    return training