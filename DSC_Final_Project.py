message=""
#function to accept the message from the user
def test_function(entry):
    print("This is the entry:",entry)
    global message
    message=entry
pos=0
neg=0
nu=0
#function to fact check
def get_news(message):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as p

    #Cleaning the texts
    import re
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.stem import WordNetLemmatizer
    from textblob import TextBlob
    lemmatizer = WordNetLemmatizer()
    corpus=[]
    bow=[]
    nltk.download('wordnet')
    review=re.sub('[^a-zA-Z]', ' ', message)
    review=review.lower()#converts all characters to lowercase
    review=review.split()#splits the sentence into a list
    lemmatizer = WordNetLemmatizer()
    review=[lemmatizer.lemmatize(word,pos="v") for word in review if not word in set(stopwords.words('english'))]# removal of stopwords
    review=' '.join(review)#converting the list back into a sentence
    corpus.append(review)#creating a list of sentences
    bow.append(review.split(" "))#creating a list of words in each sentences and storing it in a list
    bowa=review.split()
    bowb=set(bowa)
    worddict=dict.fromkeys(bowb,0)


    #SENTIMENT ANALYSIS
    
    def clean_text(inp):
        '''
        Utility function to clean text by removing links, special characters
        using simple regex statements.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", inp).split())

    def get_text_sentiment(inp):
            '''
            Utility function to classify sentiment of passed text
            using textblob's sentiment method
            '''
            analysis = TextBlob(clean_text(inp))
            if analysis.sentiment.polarity > 0:
                return 'positive'
            elif analysis.sentiment.polarity == 0:
                return 'neutral'
            else:
                return 'negative'


    def get_texts(inp):
            
            text_sentl=[]
            for t in inp:
                text_sent={}
                text_sent['text'] = t
                text_sent['sentiment'] = get_text_sentiment(t)
                text_sentl.append(text_sent)
            return text_sentl



    
    #finding the frequency of words in each sentence
    for word in bowa:
        worddict[word]+=1

    #computing the term frequency
    def computeTF(wordDict, bow):
        tfDict = {}
        bowCount = len(bow)
        for word, count in wordDict.items():
            tfDict[word] = count/float(bowCount)
        return tfDict
    tfBowA = computeTF(worddict, bowa)
    from collections import Counter
    # Initial Dictionary 
    k = Counter(tfBowA) 
    # Finding 3 highest values 
    high = k.most_common(10)  
    #print(high,"\n")
    sentence=[]
    for i in high: 
        sentence.append(i[0])


    def get_cosine_sim(*strs): 
        vectors = [t for t in get_vectors(*strs)]
        d1=np.array([vectors[0]])
        d2=np.array([vectors[1]])
        return cosine_similarity(d1,d2)
    
    def get_vectors(*strs):
        text = [t for t in strs]
        vectorizer = CountVectorizer(text)
        vectorizer.fit(text)
        return vectorizer.transform(text).toarray()

    
    
    
    #SCRAPING
    from googlesearch import search
    from newspaper import Article
    links=list()
    sentence=' '.join(sentence)
    query =sentence
    print(query)
    for j in search(query, tld="com", num=10, start=0, stop=10, pause=2.0): 
        #print(j)
        links.append(j)
    global pos
    global neg
    global nu

    #GETS THE ARTICLES FROM THEIR LINKS

    flag=0
    for k in links:
        if((k[:20]=="https://timesofindia") | (k[:18]=="https://www.news18") | (k[:26]=="https://www.hindustantimes") | (k[:21]=="https://indianexpress")\
                                           | (k[:20]=="https://www.livemint") | (k[:21]=="https://economictimes")\
                                           | (k[:22]=="https://www.indiatoday") | (k[:20]=="https://gadgets.ndtv")\
                                           | (k[:24]=="https://www.timesnownews") | (k[:19]=="https://edition.cnn")\
                                           | (k[:15]=="https://www.bbc") | ("washingtonpost" in k) | ("theguardian" in k) | ("news.com.au" in k)\
                                           | ("abc.net.au" in k) | ("www.nytimes" in k) | ("www.bloomberg" in k) | ("www.dailymail" in k)\
                                           | ("www.newyorker" in k) | ("www.mirror.co" in k) | ("www.telegraph.co" in k) | ("news.sky" in k) | ("wikipedia.org" in k)):
            #A new article from TOI
            url=k
            #For different language newspaper refer above table 
            article = Article(url, language="en") # en for English 
  
            #To download the article 
            article.download() 
 
            #To parse the article 
            article.parse()

            #To perform natural language processing ie..nlp 
            article.nlp()


            #CHECKING SENTIMENT
            temp=(article.text).split('\n')
            file=open(r"C:\Users\Saksham\Desktop\article.txt","a+",encoding="utf-8")
            file.writelines(temp)
            file=open(r"C:\Users\Saksham\Desktop\article.txt","r",encoding="utf-8")
            t=file.read()
            text=[t]
            textinp=get_texts(text)
            for i in textinp:
                print(i['sentiment'])
                if(i['sentiment']=="positive"):
                    pos=pos+1
                elif(i['sentiment']=="negative"):
                    neg=neg+1
                else:
                    nu=nu+1;
            file=open(r"C:\Users\Saksham\Desktop\article.txt","w",encoding="utf-8")

            #FINDING THE COSSIM VALUE
            message2=article.text
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            cossim=get_cosine_sim(message,message2)
            if(cossim<0.75):
                lines=message2.split('.')
                for line in lines:
                    cossim=get_cosine_sim(message,line)
                    cossim=cossim[0][0]
                    if(cossim>0.75 or cossim>0.4):
                        break
            
    if(pos>neg and pos>nu):
        sent="positive"
    elif(neg>pos and neg>nu):
        sent="negative"
    else:
        sent="neutral"


    
    if(cossim>=0.6):
        label['text']="It is true and similarity co-efficient is:",str(cossim),"sentiment is ",sent
    elif(cossim<0.6 and cossim>0.4):
        label['text']="Data is insufficient",str(cossim),"sentiment is ",sent
    else:
        label['text']="It is false and similarity co-efficient is:",str(cossim),"sentiment is ",sent
    

import tkinter as tk        


# GUI (TKINTER)
root=tk.Tk()
canvas=tk.Canvas(root,height=1000,width=1200)
canvas.pack()

background_image=tk.PhotoImage(file="C:/Users/Saksham/Downloads/image.gif")
background_label=tk.Label(root,image=background_image)
background_label.place(relwidth=1,relheight=1)

frame=tk.Frame(root, bg='#80c1ff',bd=5)
frame.place(relx=0.5,rely=0.1,relwidth=0.75,relheight=0.1,anchor='n')

entry=tk.Entry(frame,font=40)
entry.place(relwidth=0.65,relheight=1)

button=tk.Button(frame,text="GO",font=40,command= lambda: get_news(entry.get()))
button.place(relx=0.7,relwidth=0.3,relheight=1)

lower_frame=tk.Frame(root,bg='#80c1ff',bd=10)
lower_frame.place(relx=0.5,rely=0.25,relwidth=0.75,relheight=0.6,anchor='n')

label=tk.Label(lower_frame,bg='white',font=('Courier',15),anchor='nw',justify='left',bd=4)
label.place(relwidth=1,relheight=1)

root.mainloop()


