from pprint import pprint
from Parser import Parser
import util
import os
import math
import numpy as np
import argparse

class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentVectors = []

    #Mapping of vector index to keyword
    vectorKeywordIndex=[]

    vectorIDF = []
    tfidf = []
    #Tidies terms
    parser=None


    def __init__(self, documents=[]):
        self.documentVectors=[]
        self.parser = Parser()
        if(len(documents)>0):
            self.build(documents)

    def build(self,documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        self.vectorIDF = [0] * len(self.vectorKeywordIndex)
        self.documentVectors = [self.makeVector(document) for document in documents]
        #print(self.vectorIDF)
        #print(self.vectorKeywordIndex)
        #print(self.documentVectors)


    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        #Mapped documents into a single word string	
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
        
        return vectorIndex  #(keyword:position)


    def makeVector(self, wordString):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        for word in wordList:
            vector[self.vectorKeywordIndex[word]] += 1; #Use simple Term Count Model
            self.vectorIDF[self.vectorKeywordIndex[word]] += 1;
        return vector


    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        query = self.makeVector(" ".join(termList))
        return query


    def related(self,documentId):
        """ find documents that are related to the document indexed by passed Id within the document Vectors"""
        ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        return ratings

    def search(self,searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        return ratings
    
    def TFED(self, searchList):
        queryVector = self.buildQueryVector(searchList)

        ratings = [util.Euclidean_Distance(queryVector, documentVector) for documentVector in self.documentVectors]
        return ratings
    
    def IDFCOS(self, searchList):
        queryVector = self.buildQueryVector(searchList)

        self.vectorIDF = [float(2048.0/x) for x in self.vectorIDF]
        self.vectorIDF = [float(math.log10(x)) for x in self.vectorIDF]
        self.tfidf = [map(lambda (a,b):a*b,zip(self.vectorIDF, documentVector)) for documentVector in self.documentVectors]
        #print(tfidf)
        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.tfidf]
        #print(ratings)
        return ratings
    
    def IDFED(self, searchList):
        queryVector = self.buildQueryVector(searchList)

        #print(self.vectorIDF)
        #tfidf = [map(lambda (a,b):a*b,zip(self.vectorIDF, documentVector)) for documentVector in self.documentVectors]
        #print(tfidf)
        ratings = [util.Euclidean_Distance(queryVector, documentVector) for documentVector in self.tfidf]
        #print(ratings)
        return ratings


    def Feedback(self, searchList):
        queryVector = self.buildQueryVector(searchList)
        '''
        self.vectorIDF = [float(2048.0/x) for x in self.vectorIDF]
        self.vectorIDF = [float(math.log10(x)) for x in self.vectorIDF]
        self.tfidf = [map(lambda (a,b):a*b,zip(self.vectorIDF, documentVector)) for documentVector in self.documentVectors]
        '''
        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.tfidf]
        maxone = 0
        targetone = 0
        for i in range(len(ratings)):
            if (ratings[i] > maxone):
                maxone = ratings[i]
                targetone = i
        newqueryVector = []
        for i in range(len(queryVector)):
            newqueryVector.append(queryVector[i] + ((0.5) * self.documentVectors[targetone][i]))
        #print(queryVector)
        #print(newqueryVector)
        ratings2 = [util.cosine(newqueryVector, documentVector) for documentVector in self.tfidf]

        #print(ratings)
        return ratings2

if __name__ == '__main__':
    #test data
    #documents = ["The cat in the hat disabled",
    #             "A cat is a fine pet ponies.",
    #             "Dogs and cats make good pets.",
    #             "I haven't got a hat."]
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-q", "--query", help="input the query string",
                    default="drill wood sharp", required=True, nargs='+')
    args = parser.parse_args()
    queryString = ""
    count = 0
    for ch in args.query:
        if count!=0:
            queryString +=" "
            count += 1
        queryString +=ch
    find = [queryString]
    print(find)
    documents = []
    count = 0
    name = []
    #Dictionary = dict()
    yourPath = './documents'
    allFileList = os.listdir(yourPath)
    for file in allFileList:
        f = open('./documents/'+file)
        #print(file)
        name.append(int(file[:-8]))
        count += 1

        content = ""
        for line in f:
            content+=line + " "
        documents.append(content)
        #print(content)
    #print (count)

    vectorSpace = VectorSpace(documents)
    #find = ["drill wood sharp"]
    
    #Term Frequency (TF) Weighting + Cosine Similarity
    rating1 = vectorSpace.search(find)
    Dictionary1 = dict(zip(name,rating1))
    TFcos = sorted(Dictionary1.items(), key=lambda d: d[1], reverse=True)
    TFout = TFcos[:5]
    print("Term Frequency (TF) Weighting + Cosine Similarity:\n")
    for x in TFout:
        print(x[0],round(x[1], 6))

    #Term Frequency (TF) Weighting + Euclidean Distance
    rating2 = vectorSpace.TFED(find)
    Dictionary2 = dict(zip(name,rating2))
    TFED = sorted(Dictionary2.items(), key=lambda d: d[1], reverse=False)
    TFEDout = TFED[:5]
    print("Term Frequency (TF) Weighting + Euclidean Distance:\n")
    for x in TFEDout:
        print(x[0],round(x[1], 6))
    
    #TF-IDF Weighting + Cosine Similarity
    rating3 = vectorSpace.IDFCOS(find)
    Dictionary3 = dict(zip(name,rating3))
    TFIDFcos = sorted(Dictionary3.items(), key=lambda d: d[1], reverse=True)
    TFIDFcosout = TFIDFcos[:5]
    print("TF-IDF Weighting + Cosine Similarity:\n")
    for x in TFIDFcosout:
        print(x[0],round(x[1], 6))
    
    #TF-IDF Weighting + Euclidean Distance
    rating4 = vectorSpace.IDFED(find)
    Dictionary4 = dict(zip(name,rating4))
    TFIDFed = sorted(Dictionary4.items(), key=lambda d: d[1], reverse=False)
    TFIDFedout = TFIDFed[:5]
    print("TF-IDF Weighting + Euclidean Distance:\n")
    for x in TFIDFedout:
        print(x[0],round(x[1], 6))
    
    #Relevance Feedback
    rating5 = vectorSpace.Feedback(find)
    Dictionary5 = dict(zip(name,rating5))
    Feedback = sorted(Dictionary5.items(), key=lambda d: d[1], reverse=True)
    FeedBackout = Feedback[:5]
    print("FeedBack Queries + TF-IDF Weighting + Cosine Similarity:\n")
    for x in FeedBackout:
        print(x[0],round(x[1], 6))



    #print(vectorSpace.vectorKeywordIndex)
    #print(vectorSpace.documentVectors)
    #print(vectorSpace.related(1))
    #print(vectorSpace.search(["drill wood sharp"]))

###################################################
