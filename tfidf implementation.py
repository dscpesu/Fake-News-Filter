docA = "The cat sat on my face"
docB = "The dog sat on my bed"
bowA = docA.split(" ")#splitting the word 
bowB = docB.split(" ")
print(bowB)
wordSet = set(bowA).union(set(bowB))#taking a set of all the words in both the sentences 
print(wordSet)
wordDictA = dict.fromkeys(wordSet, 0) #creating a dictionary that shows the frequency of the words in each sentence
wordDictB = dict.fromkeys(wordSet, 0)
print(wordDictA)
print(wordDictB)
for word in bowA: #finding the frequency of words in each sentence
    wordDictA[word]+=1
    
for word in bowB:
    wordDictB[word]+=1
print(wordDictB)

def computeTF(wordDict, bow):#computing the term frequency
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict
tfBowA = computeTF(wordDictA, bowA)
tfBowB = computeTF(wordDictB, bowB)
print(tfBowA)
print(tfBowB)
def computeIDF(docList):#computing the inverse document frequency
    import math
    idfDict = {}
    N = len(docList)
    
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))
        
    return idfDict
idfs = computeIDF([wordDictA, wordDictB])
def computeTFIDF(tfBow, idfs):#computing the tfidf score
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf
tfidfBowA = computeTFIDF(tfBowA, idfs)
tfidfBowB = computeTFIDF(tfBowB, idfs)
import pandas as pd #displaying the words with the scores in a table
pd.DataFrame([tfidfBowA, tfidfBowB])