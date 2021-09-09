# from google.colab import drive
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# drive.mount("/content/gdrive")

# folder_location = "/content/gdrive/MyDrive/MovieLens/"
folder_location = "Dataset/"



# users.dat > user_id | gender | age |  ??
# movies.dat > movie_id | title | genres
# ratings.dat > user_id | movie_id | rating | timestamp

users = pd.read_csv( folder_location + "users.dat", delimiter="::")
movies = pd.read_csv( folder_location + "movies.csv")
ratings = pd.read_csv( folder_location + "ratings.csv", nrows=100000)
links = pd.read_csv( folder_location + "links.csv", dtype={'imdbId': str })



movies["release_date"] = 0
movies["cover"] = 0
movies.to_csv(folder_location + "movies.csv", index=False)



user_rating_matrix = pd.pivot_table(ratings, values="rating", index="userId",
                    columns="movieId")           
user_rating_matrix = user_rating_matrix.fillna(0)





# GetSimilarUsers( dataframe,
#                  method-to-group-similar-users, 
#                  user-id of user we are finding matches for, 
#                  number of similar users to find)
def GetSimilarUsers(_dframe, method="cosine", user_id=0, n=10):
  
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


  similarity =  similarity[0 : 50]

  
  return similarity
  
  #print(similarity.head(50))
  #print(movies[movies.movie_id.isin(similarity.movie_id)].head())

#GetSimilarMovies(3448)
  



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

def DownloadBatch(dataset):
  dataset.reset_index(inplace=True)


  for i in range(0, len(dataset) ):

    if(movies.loc[movies.movieId == dataset.loc[i, "movieId"],"cover"].item() == 0 ):
      val = tmdbAPI.GetMovieCover(dataset.iloc[i]["movieId"].astype(int))
      dataset.loc[i, "cover"] = val
      movies.loc[movies.movieId == dataset.loc[i, "movieId"],"cover"] = val
    else:
      dataset.loc[i, "cover"]  = movies.loc[movies.movieId == dataset.loc[i, "movieId"],"cover"].item()
  movies.to_csv(folder_location + "movies.csv", index=False)




def DownloadBatch_ReleaseDate(dataset):
  dataset.reset_index(inplace=True)

  for i in range(0, len(dataset) ):
    if(movies.loc[movies.movieId == dataset.loc[i, "movieId"], "release_date"].item() == 0 ):
      val =  int(tmdbAPI.GetReleaseDate(dataset.loc[i, "movieId"]))
      dataset.loc[i, "release_date"] = val
      movies.loc[movies.movieId == dataset.loc[i, "movieId"], "release_date"] = val
    else:
      dataset.loc[i, "release_date"] = movies.loc[movies.movieId == dataset.loc[i, "movieId"], "release_date"].item()
  movies.to_csv(folder_location + "movies.csv", index=False)


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

def AddReleaseDate(movies):
  #set full  Release Date column empty  
  #movies["release_date"] = 0;

  #Get the release date from title and assign that release date to its respective column 
  #for i in range(0, len(movies), int(len(movies))):
    #th = threading.Thread(target=DownloadBatch_ReleaseDate,args=( i , int(i + 1000)) )
    #th.start()
    #movies.loc[i, "release_date"] = tmdbAPI.GetReleaseDate(movies.iloc[i].movieId.item())
  DownloadBatch_ReleaseDate(0, len(movies) )
# if 1:
#   AddReleaseDate(movies)


if False:
  print( "Total movie dataset len is : ", len(movies))
  MakeImageDataSet()
# DownloadBatch(0,10)
movies.fillna("", inplace=True)
#imdbAPI.GetMovieCover(1)


from flask import Flask, request, jsonify


# class WebServer:
  
#   port = 8080
#   app = None
#   def __init__(self, port):
#     self.app = Flask(__name__)
  
#   @self.app.route("/movies/filter/collab")
#   def GetCollabFiltering():
#     body = request.get_json()
#     if("user" in body.keys()):
#       user = body["user"]


import json

app = Flask(__name__)

from flask_cors import CORS

CORS(app)





@app.route("/movies/filter/collab", methods=["POST"])
def GetCollabFiltering():
  body = request.get_json()
  if(body is not None and "user" in body.keys()):
    user = body["user"]


    similarity_index = GetSimilarUsers(user_rating_matrix, "cosine", user, 20)
    recommended_movies  = RecommendMoviesByRating(similarity_index, user_rating_matrix, movies, 50)
    recommended_movies = recommended_movies.to_frame()
    recommended_movies = recommended_movies.merge(movies, left_index=True, right_index=True)

    print(recommended_movies.head())

    DownloadBatch(recommended_movies)
    DownloadBatch_ReleaseDate(recommended_movies)
    recommended_movies.fillna(0)
    _data = (recommended_movies.to_dict( orient='records'))
   # print({ "message" : "success" , "data" : _data })
    return json.dumps({ "message" : "success" , "data" : _data }), 200, {'Content-Type': 'application/json; charset=utf-8' }
  else:
    return { "error" : "User is not defined in request."} , 400


@app.route("/movies/filter/similar", methods=["POST"])

def GetSimilarItemFiltering():
  body = request.get_json()
  if(body is not None and "user" in body.keys()):
    user = body["user"]

    randomUserRating = ratings[ratings.userId == user].sample(n=1)
    baseMovie = movies[movies.movieId == randomUserRating.movieId.item()]
    similarMovies = GetSimilarMovies(randomUserRating.movieId.item(), movies)

    print(similarMovies.head())

    DownloadBatch(similarMovies)
    DownloadBatch_ReleaseDate(similarMovies)
    baseMovie = baseMovie.to_dict(orient="records")
    similarMovies.fillna(0)

    _data = (similarMovies.to_dict( orient='records'))
    #print({ "message" : "success" , "data" : _data })
    return json.dumps({ "message" : "success" , "base_movie" : baseMovie , "data" : _data }), 200, {'Content-Type': 'application/json; charset=utf-8' }
  else:
    return { "error" : "User is not defined in request."} , 400
if __name__ == "__main__":
    app.run(debug=True)