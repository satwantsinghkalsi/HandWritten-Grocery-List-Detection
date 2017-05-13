import pickle
#import numpy as np
#f= open('chars_cls.pkl', 'rb')
#v = cPickle.load(f)
#data = np.load(f)
#print (data)
with open('chars_cls.pkl','rb') as f:
     print(pickle.load(f))
#f=open('chars_cls.pkl','rb')
#pickle.load(f)
#pickle.dump(w, open("a_py2.pkl","wb"), protocol=2)
