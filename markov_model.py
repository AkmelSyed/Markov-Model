import numpy as np
import pandas as pd

class MarkovModel:
    def __init__(self, order=1):
        self.window = order+1
        
    def change_type(self, record):
        if(np.isin(record.dtype, ['<U1','<U2','<U3','<U4','<U5','<U6','<U7','<U8','<U9'])):
            record = record.astype('<U10')
        else:
            record = record.astype(str)
            
        return record
        
    def prepare_record(self, record):
        record = self.change_type(np.array(record))
        record = np.insert(record, 0, '***None***')
        record = np.append(record, '***None***')
        
        return record        
    
    def create_grams_list(self, tokenized):
        grams_list = []
                        
        for record in tokenized:
            record = self.prepare_record(record)
            cols = record.shape[0]
            for i in range(cols):
                pairings = record[i:i+self.window]
                if(len(pairings) == self.window):
                    grams_list.append(pairings)
                    
        return grams_list
    
    
    def fit(self, tokenized):        
        grams_list = self.create_grams_list(tokenized)
        df = pd.DataFrame(grams_list)
        _, cols = df.shape
        
        precondition = ''
        for i in range(cols-1):
            precondition += df[i].map(str) + ' '       
        
        self.markov_table = pd.DataFrame({'Precondition': precondition.str.strip(), 'Prediction': df[df.columns[-1]]})
    
    
    def predict(self, X):
        given = str(X)
        subset = self.markov_table[self.markov_table['Precondition'] == given]
        rows, _ = subset.shape
        random_guess = np.random.randint(rows)
        return subset['Prediction'].iloc[random_guess]
    
    
    def continuously_predict(self, starting='***None***', how_many=1):
        for i in range(how_many):
            word = ' '.join(starting.split(' ')[-self.window+1:])
            
            ###Check to see if order of words existed in the training
            try:
                predicted = self.predict(word)
            except:
            ###If word order of words didn't exist in the trained set, then break out
                break                    
            
            starting += ' ' + predicted
            
        return starting
        
    
    
mm = MarkovModel()

sentences = [
            "Today you are you. That is truer than true. There is no one alive who is you-er than you.".split(' '),
            "You have brains in your head. You have feet in your shoes. You can steer yourself any direction you choose. You’re on your own.".split(' '),
            "The more that you read, the more things you will know. The more that you learn, the more places you’ll go.".split(' '),
            "Think left and think right and think low and think high. Oh, the thinks you can think up if only you try.".split(' ')
            ]

###must pass a list of lists
mm.fit(sentences)

mm.continuously_predict(starting='***None***', how_many=100)