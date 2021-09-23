# from google.colab import drive
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from Model.model import ParseGenre, WreckEm


folder_location = "Dataset/"

# LOAD CSV FILES

users = pd.read_csv( folder_location + "users.dat", delimiter="::")
movies = pd.read_csv( folder_location + "movies_training.csv")
ratings = pd.read_csv( folder_location + "ratings_training.csv", nrows=1000000)
links = pd.read_csv( folder_location + "links.csv", dtype={'imdbId': str })


print("Movies Data-Set len is ", len(movies))
print("Total unique users ", len(ratings.userId.unique()))



#LOAD PYTHON MODEL

model_location = "Model/model_rev1.pth"

usersLen = int(ratings.userId.max()) + 1
moviesLen = int(ratings.movieId.max()) + 1

model = WreckEm(moviesLen, usersLen)

try:
  model.load_state_dict(torch.load(model_location))
except FileNotFoundError as e:
  print("No save file found.")

model.eval()




# User x Movies - Matrix

user_rating_matrix = pd.pivot_table(ratings, values="rating", index="userId",
                    columns="movieId")           
user_rating_matrix = user_rating_matrix.fillna(0)





# GetSimilarUsers( dataframe,
#                  method-to-group-similar-users, 
#                  user-id of user we are finding matches for, 
#                  number of similar users to find)
def GetSimilarUsers(_dframe, method="cosine", user_id=0, n=200):
  
  user = _dframe[_dframe.index == user_id]
  otherUsers = _dframe[_dframe.index != user_id]

  similarity = None



  similarity = cosine_similarity(user, otherUsers)[0]


  similarity = pd.DataFrame({ "userId" : otherUsers.index, "similarity" : similarity  }) 

  similarity = similarity.sort_values(by=['similarity'], ascending=False)


  return similarity.iloc[0:n]

def RecommendMoviesByRating(similarity_index, user_rating_matrix, movies, n):

  similar_users = user_rating_matrix[user_rating_matrix.index.isin(similarity_index.userId)]
  
  mean_movie_rating = (similar_users.mean(axis=0))
  mean_movie_rating = mean_movie_rating[mean_movie_rating != 0] 
  similar_movies = (mean_movie_rating.sort_values( ascending=False)).iloc[0:n]

  return similar_movies

#users_highest_rated_movies = (ratings[ratings.user_id == 24].sort_values(by=["rating"], ascending=False).movie_id.reset_index(drop=True))





#print(movies.head())

#################################################
#            CONTENT BASED FILTERING            #
#################################################





def GetSimilarMovies(movieId, movies):
  #movieId = 4896
  movieAndLabelMatrix = { "movieId" : [] }

  for i in range(0, len(movies)):
    genres = movies.genres.iloc[i]
    genres = genres.split("|")
    for g in genres:
      if g not in movieAndLabelMatrix:
        movieAndLabelMatrix[g] = []
  for i in range(0, len(movies)):
    movie_id = movies.movieId.iloc[i]
    genres = movies.genres.iloc[i]
    genres = genres.split("|")
    for k in movieAndLabelMatrix.keys():
      movieAndLabelMatrix[k].append(0)
    movieAndLabelMatrix["movieId"][-1] = movie_id
    for g in genres:
      movieAndLabelMatrix[g][-1] = 1

  movieMatrix = pd.DataFrame(movieAndLabelMatrix)

  _movie = movieMatrix[movieMatrix.movieId == movieId]

  _otherMovies = movieMatrix[movieMatrix.movieId != movieId]

  _movie.set_index("movieId", inplace=True)
  _otherMovies.set_index("movieId", inplace=True)

  #print(_movie.head())

  #print(_otherMovies.head())

  similarity = cosine_similarity(_movie, _otherMovies)[0]
  similarity = pd.DataFrame({ "movieId" : _otherMovies.index, "similarity" : similarity  }) 
  similarity = similarity.sort_values(by=['similarity'], ascending=False).merge(movies)

  referenceMovie = movies[movies.movieId == movieId]

  #similarity.loc[referenceMovie["popular_cast"].item() == similarity.popular_cast, "similarity"] *= 1.25

  for i in range(0, len(similarity)):
    if(referenceMovie["popular_cast"].item() == similarity.iloc[i]["popular_cast"]):
      similarity.loc[i,"similarity"] *= 2
  similarity = similarity.sort_values(by=["similarity"], ascending=False)
  similarity =  similarity[0 : 50]  
  return similarity
  
  #print(similarity.head(50))
  #print(movies[movies.movie_id.isin(similarity.movie_id)].head())

#GetSimilarMovies(3448)
def GetFavoriteActors(userId):
  moviesRatedByUser = ratings[ratings.userId == userId]
  actorRating = {}
  moviesRatedByUser = moviesRatedByUser.dropna();
  for i in range(0, len(moviesRatedByUser)):
    try:
      if(movies[movies.movieId == moviesRatedByUser.iloc[i]["movieId"].item()]["popular_cast"].item() != np.nan):
        if(movies[movies.movieId == moviesRatedByUser.iloc[i]["movieId"].item()]["popular_cast"].item() in actorRating.keys()):
          actorRating[movies[movies.movieId == moviesRatedByUser.iloc[i]["movieId"].item()]["popular_cast"].item()].append(moviesRatedByUser.iloc[i]["rating"].item())
        else:
          
          actorRating[movies[movies.movieId == moviesRatedByUser.iloc[i]["movieId"]]["popular_cast"].item()] = []
          actorRating[movies[movies.movieId == moviesRatedByUser.iloc[i]["movieId"]]["popular_cast"].item()].append(moviesRatedByUser.iloc[i]["rating"].item())
    except Exception as e:
      print("Error", movies[movies.movieId == moviesRatedByUser.iloc[i]["movieId"].item()]["popular_cast"].item())
  actorRatingDf = { "actor" : [] , "rating" : []}
  for key in actorRating.keys():
    actorRatingDf["actor"].append(key)
    
    averageRating = (sum(actorRating[key]) / len(actorRating[key])) * (1.025 ** len(actorRating[key]))
    # print(len(actorRating[key]))s
    actorRatingDf["rating"].append(averageRating)
  actorRatingDf = pd.DataFrame(actorRatingDf)
  actorRatingDf.dropna(inplace=True)
  actorRatingDf = actorRatingDf.sort_values(by=["rating"], ascending=False)
  return actorRatingDf

print(GetFavoriteActors(99).head(100))


from bs4 import BeautifulSoup
import requests, json
class TMDB_API:

  API_KEY = "2e992aa6b414d023e0c421a5619df5b7"
  movieID_imdbID_table = None
  
  def __init__(self):
    return
  
  def SetDataSet(self, dataset):
    self.movieID_imdbID_table = dataset
  
  
  def GetImdbID(self, movieId):

    #print(self.movieID_imdbID_table[self.movieID_imdbID_table.movieId == movieId])
    return self.movieID_imdbID_table[self.movieID_imdbID_table.movieId == movieId]["tmdbId"].item()
  
  def GetMovieCover(self, movieId):
    tmdbId = self.GetImdbID(movieId)

    try:

      req = requests.get("https://api.themoviedb.org/3/movie/"  + str(tmdbId) +  "/images?api_key="  + self.API_KEY + "&language=en")
     # print("https://api.themoviedb.org/3/movie/"  + str(int(tmdbId)) +  "/images?api_key="  + self.API_KEY + "&language=en")
      if req.status_code == 200:
        
        tmdbData = json.loads(req.text)
        
        if("posters" in tmdbData and len(tmdbData["posters"]) > 0):
          #print(tmdbData)
          
          return "https://image.tmdb.org/t/p/w500" + tmdbData["posters"][0]["file_path"]
        # imdbPage = req.text
        # imdbSoup = BeautifulSoup(imdbPage, 'html.parser')
        # coverImage = imdbSoup.find_all("img", { "class"  : "ipc-image" })

        # if coverImage != None:
        #   print(coverImage[0]["src"])
        #   return coverImage[0]["src"]
        # else:
        #   print("Some error occured extracting cover image from source url." )
      else:
        #print(req.status_code)
        print("Some error occured getting data from tmdb api. ID ->", str(tmdbId) )
    except Exception as e:
      print("Some internal error occured. ", e )


  def GetReleaseDate(self, movieId):
    


    try:
      tmdbId = int(self.GetImdbID(movieId))

      req = requests.get("https://api.themoviedb.org/3/movie/"  + str(tmdbId) +  "?api_key="  + self.API_KEY)
      #print("https://api.themoviedb.org/3/movie/"  + str(tmdbId) +  "/?api_key="  + self.API_KEY )
     # print("https://api.themoviedb.org/3/movie/"  + str(int(tmdbId)) +  "/images?api_key="  + self.API_KEY + "&language=en")
      if req.status_code == 200:
        
        tmdbData = json.loads(req.text)
        
        if("release_date" in tmdbData ):
          #print(tmdbData)
          
          return int(tmdbData["release_date"].split("-")[0])
        # imdbPage = req.text
        # imdbSoup = BeautifulSoup(imdbPage, 'html.parser')
        # coverImage = imdbSoup.find_all("img", { "class"  : "ipc-image" })

        # if coverImage != None:
        #   print(coverImage[0]["src"])
        #   return coverImage[0]["src"]
        # else:
        #   print("Some error occured extracting cover image from source url." )
      else:
        #print(req.status_code)
        print("Some error occured getting data from tmdb api. ID ->", str(tmdbId), req.status_code )
        return 1990
    except Exception as e:
      print("Some internal error occured. ", e )
      return 1990
  
  def GetMovieDescription(self, movieId):
    return




import threading, time
import numpy as np

tmdbAPI = TMDB_API()
tmdbAPI.SetDataSet(links)

# def DownloadBatch(dataset):
#   dataset.reset_index(inplace=True)


#   for i in range(0, len(dataset) ):

#     if(movies.loc[movies.movieId == dataset.loc[i, "movieId"],"cover"].item() == 0 ):
#       val = tmdbAPI.GetMovieCover(dataset.iloc[i]["movieId"].astype(int))
#       dataset.loc[i, "cover"] = val
#       movies.loc[movies.movieId == dataset.loc[i, "movieId"],"cover"] = val
#     else:
#       dataset.loc[i, "cover"]  = movies.loc[movies.movieId == dataset.loc[i, "movieId"],"cover"].item()
#   movies.to_csv(folder_location + "movies.csv", index=False)




# def DownloadBatch_ReleaseDate(dataset):
#   dataset.reset_index(inplace=True)

#   for i in range(0, len(dataset) ):
#     if(movies.loc[movies.movieId == dataset.loc[i, "movieId"], "release_date"].item() == 0 ):
#       val =  int(tmdbAPI.GetReleaseDate(dataset.loc[i, "movieId"]))
#       dataset.loc[i, "release_date"] = val
#       movies.loc[movies.movieId == dataset.loc[i, "movieId"], "release_date"] = val
#     else:
#       dataset.loc[i, "release_date"] = movies.loc[movies.movieId == dataset.loc[i, "movieId"], "release_date"].item()
#   movies.to_csv(folder_location + "movies.csv", index=False)


def MakeImageDataSet():
  lastindex = 0
  for i in range(0, len(movies), 1):
    # th = threading.Thread(target=DownloadBatch,args=( i , i + 1024) )
    # print(i/1024)
    # th.start()
    try:
      if(not pd.isna(movies.loc[i,"cover"]) ):
        
        print("Skipping ", i + 1, movies.loc[i,"cover"])
      else:
        movies.loc[i, "cover"] = tmdbAPI.GetMovieCover(movies.loc[i, "movieId"].astype(int))
        print(movies.loc[i, "cover"])
    except KeyError as err:
      movies.loc[i, "cover"] = tmdbAPI.GetMovieCover(movies.loc[i, "movieId"].astype(int))
      print(movies.loc[i, "cover"])

    if(i % 10 == 0):
      movies.to_csv("movies.csv", index=False)

# def AddReleaseDate(movies):
#   #set full  Release Date column empty  
#   #movies["release_date"] = 0;

#   #Get the release date from title and assign that release date to its respective column 
#   #for i in range(0, len(movies), int(len(movies))):
#     #th = threading.Thread(target=DownloadBatch_ReleaseDate,args=( i , int(i + 1000)) )
#     #th.start()
#     #movies.loc[i, "release_date"] = tmdbAPI.GetReleaseDate(movies.iloc[i].movieId.item())
#   DownloadBatch_ReleaseDate(0, len(movies) )
# # if 1:
# #   AddReleaseDate(movies)


# if False:
#   print( "Total movie dataset len is : ", len(movies))
#   MakeImageDataSet()


movies.fillna("", inplace=True)
ratings.fillna("", inplace=True)


from flask import Flask, request, jsonify




import json

app = Flask(__name__)

from flask_cors import CORS

CORS(app)





@app.route("/movies/filter/collab", methods=["POST"])
def GetCollabFiltering():
  body = request.get_json()
  if(body is not None and "user" in body.keys()):
    user = body["user"]


    similarity_index = GetSimilarUsers(user_rating_matrix, "cosine", user, 100)
    recommended_movies  = RecommendMoviesByRating(similarity_index, user_rating_matrix, movies, 50)
    recommended_movies = recommended_movies.to_frame()
    recommended_movies = recommended_movies.merge(movies, left_index=True, right_index=True)

    print(similarity_index.head(10))

    recommended_movies.fillna(-1)
    _data = (recommended_movies.to_dict( orient='records'))

    return json.dumps({ "message" : "success" , "data" : _data }), 200, {'Content-Type': 'application/json; charset=utf-8' }
  else:
    return { "error" : "User is not defined in request."} , 400


@app.route("/movies/filter/similar", methods=["POST"])

def GetSimilarItemFiltering():
  body = request.get_json()
  if(body is not None and "user" in body.keys()):
    user = body["user"]

    randomUserRating = ratings[ratings.userId == user].sample(n=1)
    baseMovie = movies[movies.movieId == randomUserRating.movieId.item()] #Original
    similarMovies = GetSimilarMovies(randomUserRating.movieId.item(), movies) #Original

    # baseMovie = movies[movies.movieId == 4896]

    # similarMovies = GetSimilarMovies(4896, movies)

    print(similarMovies.head())

    baseMovie = baseMovie.to_dict(orient="records")
    similarMovies.fillna(-1)

    _data = (similarMovies.to_dict( orient='records'))

    return json.dumps({ "message" : "success" , "base_movie" : baseMovie , "data" : _data }), 200, {'Content-Type': 'application/json; charset=utf-8' }
  else:
    return { "error" : "User is not defined in request."} , 400

@app.route("/movies/filter/ai", methods=["POST"])
def AiFiltering():
  body = request.get_json()
  if(body is not None and "user" in body.keys()):
    user = body["user"]
    moviesWatchedByUser = ratings[ratings.userId == user].movieId
    
    moviesUnwatched = movies[~movies.movieId.isin(moviesWatchedByUser)]
    moviesUnwatched = moviesUnwatched.sample(n=1000)

    data = []
    model.eval()
    for i in range(0, len(moviesUnwatched) - 1):
      userId = torch.LongTensor([ user ])
      movieId = torch.LongTensor([ moviesUnwatched.iloc[i]["movieId"].item() ])
      genre = torch.Tensor([ParseGenre(moviesUnwatched.iloc[i]["genres"])])
      vote_average = torch.Tensor([ [  moviesUnwatched.iloc[i]["vote_average"].item()/ 10.0  ] ])
      release_date = None
      
      recommendation = model(userId, movieId, genre, vote_average, release_date)
      
      data.append( {
            "0": recommendation.item(),
            "movieId": moviesUnwatched.iloc[i].movieId.item(),
            "title": moviesUnwatched.iloc[i].title,
            "genres": moviesUnwatched.iloc[i].genres,
            "release_date": moviesUnwatched.iloc[i].release_date.item(),
            "vote_average": moviesUnwatched.iloc[i].vote_average.item(),
            "revenue": moviesUnwatched.iloc[i].revenue.item(),
            "popular_cast": moviesUnwatched.iloc[i].popular_cast,
            "cover": moviesUnwatched.iloc[i].cover
        })
    data = sorted(data, key = lambda i: i['0'])
    return json.dumps({ "message" : "success" , "data" : data }), 200, {'Content-Type': 'application/json; charset=utf-8' }




if __name__ == "__main__":
    app.run(debug=True)

#Chi Square
#Feather format
#Feedback