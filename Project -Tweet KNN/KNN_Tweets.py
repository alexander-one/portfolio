
# Dependencies
import pandas as pd
import numpy as np
import nltk
import random
#nltk.download()

class KNN():

    def get_data(self):
        url = 'https://raw.githubusercontent.com/alexander-one/test/main/nytimeshealth.txt'
        self.df = pd.read_csv(url, on_bad_lines = 'skip')

    def remove_at(self):
        self.df['tweets'] = self.df['tweets'].str.replace(r'@[a-zA-Z0-9_]*', '', regex = True)
        return

    def remove_retweet(self):
        self.df['tweets'] = self.df['tweets'].str.replace(r'(RT)', '', regex = True)
        return

    def remove_timestamp(self):
        self.df['tweets'] = self.df['tweets'].str.replace(r'\|.*\|', '', regex = True)
        return

    def remove_id(self):
        self.df['tweets'] = self.df['tweets'].str.replace(r'[0-9]{18}', '', regex = True)
        return

    def remove_url(self):
        self.df['tweets'] = self.df['tweets'].str.replace(r'http\S+', '', regex = True)
        return

    def remove_punctuation(self):
        self.df['tweets'] = self.df['tweets'].str.replace(r'[^\w\s]', '', regex = True)
        return

    def tokenize(self):
        self.df['tweets'] = self.df.apply(lambda row: nltk.word_tokenize(row['tweets']), axis=1)
        return

    

    def calc_jaccard_distance(self, A, B):
        #sum the number of unique words
        #1 - number of shared words/number of unique words

        list_of_words = []
        shared_words = 0
        for word in A:
            list_of_words.append(word)
            for token in B:
                list_of_words.append(token)
                if word == token:
                    shared_words += 1


        unique_words = len(set(list_of_words))

        if unique_words != 0:
            jaccard_distance_value = 1-(shared_words/unique_words)
        else: 
            return 1

        return jaccard_distance_value

    def fill_jaccard_matrix(self):

        #for each row in the tweets column, fill the tweets_similarity matrix 
        #with Jaccard distance for every other row
        
        for row in range(len(self.df['tweets'])):
            for other_rows in range(row,len(self.df['tweets'])):
                self.tweet_similarity[row][other_rows] = self.calc_jaccard_distance(self.df['tweets'][row], self.df['tweets'][other_rows])
                self.tweet_similarity[other_rows][row] = self.tweet_similarity[row][other_rows]

        return

    def initial_centroids(self):
        
        #choose K random rows to be the initial tweets
        self.tweet_centroids = []
        keep_checking = True
        for x in range(self.number_of_clusters):

            keep_checking = True

            while keep_checking == True:

                number = random.randrange(len(self.df))

                if number in self.tweet_centroids:
                    keep_checking = True
                else:
                    self.tweet_centroids.append(number)
                    keep_checking = False

        return 

    def assign_initial_clusters(self):

        #create empty label column in DF to keep track of clusters
        cluster = np.zeros(len(self.df))
        self.df['cluster_'+str(1)] = cluster

        #for each tweet, pull jaccard distance to the clusters from the matrix [tweet][centroid]
        #the centroid is the randomly chosen row
        #in the event that all Jaccard distances are equal, it will choose the first centroid in the list

        
        for row in range(len(self.df)):
            closest_centroid = self.tweet_centroids[0]
            centroid_distance = 1
            tmp_distance = 0
            for centroid in self.tweet_centroids:

                tmp_distance = self.tweet_similarity[row][centroid]

                if tmp_distance < centroid_distance: 

                    #closest centroid is assigned
                    closest_centroid = centroid
                    #centroid_distance becomes the new low threshold needed to change cluster assignment
                    centroid_distance = tmp_distance

            self.df['cluster_'+str(1)][row] = closest_centroid

        return

    def update_cluster(self):

        for iter in range(1,self.iterations):

            tmp_tweet_centroids = []

            #for each set of tweets in one cluster, determine the one that is closest to all the other tweets in the cluster        
            for centroid in self.tweet_centroids:

                #store all the tweet row numbers in a list for all that match the same cluster
                tweets_in_cluster = []

                for row in range(len(self.df)):

                    if self.df['cluster_'+str(iter)][row] == centroid:
                        tweets_in_cluster.append(row)
            
                #this will become the new cluster
                sum_distance_tweets_cluster = 0
                distances = []
                for tweet in tweets_in_cluster:
                    for other_tweets in tweets_in_cluster:
                        sum_distance_tweets_cluster += self.tweet_similarity[tweet][other_tweets]
                    distances.append(sum_distance_tweets_cluster)
                #this will grab the index from all the tweet in cluster that matches the lowest sum distance to all other tweets in the cluster
                tmp_tweet_centroids.append(tweets_in_cluster[distances.index(min(distances))])

       
            #this will become the new cluster
            #update cluters array
            self.tweet_centroids = tmp_tweet_centroids
            #recalculate the closest centroid to each row

            #create empty label column in DF to keep track of clusters
            cluster = np.zeros(len(self.df))
            self.df['cluster_'+str(iter+1)] = cluster

            #for each tweet, pull jaccard distance to the clusters from the matrix [tweet][centroid]
            #the centroid is the randomly chosen row
            #in the event that all Jaccard distances are equal, it will choose the first centroid in the list

        
            for row in range(len(self.df)):
                closest_centroid = self.tweet_centroids[0]
                centroid_distance = 1
                tmp_distance = 0
                for centroid in self.tweet_centroids:

                    tmp_distance = self.tweet_similarity[row][centroid]

                    if tmp_distance < centroid_distance: 

                        #closest centroid is assigned
                        closest_centroid = centroid
                        #centroid_distance becomes the new low threshold needed to change cluster assignment
                        centroid_distance = tmp_distance

                self.df['cluster_'+str(iter+1)][row] = closest_centroid

        return

    def calc_sse_and_count_tweets(self):

        
        self.sse = 0
        self.number_of_tweets = []
        for centroid in self.tweet_centroids:

                #store all the tweet row numbers in a list for all that match the same cluster
                tweets_in_cluster = []
                
                for row in range(len(self.df)):

                    if self.df['cluster_'+str(self.iterations)][row] == centroid:
                        tweets_in_cluster.append(row)
            
                #this will become the new cluster
                sum_distance_tweets_cluster = 0

                for tweet in tweets_in_cluster:
                        sum_distance_tweets_cluster += self.tweet_similarity[tweet][centroid]**2
                
                number_of_tweets_in_cluster = len(tweets_in_cluster)
                self.sse += sum_distance_tweets_cluster
                self.number_of_tweets.append(number_of_tweets_in_cluster)
            
        return

    def print_stats(self):

        print("Number of Clusters: ", self.number_of_clusters)
        print("SSE: ", self.sse)

        print("Tweets in clusters: ")
        print(self.number_of_tweets)
        return

    def set_parameters(self, K, iterations):

        self.number_of_clusters = K
        self.iterations = iterations

        print("Got the data for these clusters: ", self.number_of_clusters)

        return

    def run_program(self):
        

        #choose K number of random tweets to start with
        self.initial_centroids()

        #assign clusters
        self.assign_initial_clusters()

        print("Updating clusters...")

        #update clusters
        self.update_cluster()

        self.calc_sse_and_count_tweets()

        self.print_stats()

        return

    def set_jaccard(self):
        print("Filling the Jaccard matrix...")
        #instantiate a matrix for the Jaccard distance
        #since 1 is the least connected in Jaccard, make a square matrix of 1s from the tweets
        self.tweet_similarity = np.ones((len(self.df), len(self.df)))
        self.fill_jaccard_matrix()

        return

    def __init__(self):

        self.get_data()

       

        #clean the data, this is the suggested order
        self.remove_id()
        self.remove_timestamp()
        self.remove_retweet()
        self.remove_at()
        self.remove_url()
        self.remove_punctuation()

        print("Cleaned and now tokenizing...")
        #tokenize data
        self.tokenize()
        
        return

test = KNN()
test.set_jaccard()
test.set_parameters(2,5)
test.run_program()
test.set_parameters(5,5)
test.run_program()
test.set_parameters(10,5)
test.run_program()
test.set_parameters(20,5)
test.run_program()
test.set_parameters(50,5)
test.run_program()
test.set_parameters(100,5)
test.run_program()
