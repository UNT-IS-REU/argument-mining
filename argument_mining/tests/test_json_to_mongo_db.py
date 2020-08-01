# Author: haihua
# 2020-05-09
# test json into mongoDB

from pymongo import MongoClient


def connectMongoDB():
    client = MongoClient()
    client = MongoClient("localhost", 27017)
    client = MongoClient('mongodb://localhost:27017/')
    db = client['testmongodb']
    article = {"author": "Derrick Mwiti",
               "about": "Introduction to MongoDB and Python",
               "tags":
                   ["mongodb", "python", "pymongo"]}
    articles = db.articles
    result = articles.insert_one(article)
    print("First article key is: {}".format(result.inserted_id))
    print(db.list_collection_names())

    article1 = {"author": "Emmanuel Kens",
                "about": "Knn and Python",
                "tags":
                    ["Knn", "pymongo"]}
    article2 = {"author": "Daniel Kimeli",
                "about": "Web Development and Python",
                "tags":
                    ["web", "design", "HTML"]}
    new_articles = articles.insert_many([article1, article2])
    print("The new article IDs are {}".format(new_articles.inserted_ids))
    print(articles.find_one())
    for article in articles.find():
        print(article)

if __name__ == '__main__':
    connectMongoDB()