import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np

import torch.nn as nn
import torch.nn.functional as F





s = requests.Session()

retries = Retry(total=20,
                backoff_factor=0.1,
                status_forcelist=[ 500, 502, 503, 504 ])
s.mount('https://', HTTPAdapter(max_retries=retries))


folder_location = "NewDataset/"


# users.dat > user_id | gender | age |  ??
# movies.dat > movie_id | title | genres
# ratings.dat > user_id | movie_id | rating | timestamp


movies = pd.read_csv( folder_location + "movies.csv")
links = pd.read_csv( folder_location + "links.csv", dtype={'tmdbId': str })
print("Movies len is ", len(movies))

API_KEY = "2e992aa6b414d023e0c421a5619df5b7"

def GetImdbID(movieId):
    return links[links.movieId == movieId]["tmdbId"].item()


def GetMovieCover( movieId):
    tmdbId = GetImdbID(movieId)

    try:

      req = s.get("https://api.themoviedb.org/3/movie/"  + str(tmdbId) +  "/images?api_key="  + API_KEY + "&language=en")

      if req.status_code == 200:
        
        tmdbData = json.loads(req.text)
        
        if("posters" in tmdbData and len(tmdbData["posters"]) > 0):
          #print(tmdbData)
          
          return "https://image.tmdb.org/t/p/w500" + tmdbData["posters"][0]["file_path"]

      else:
        #print(req.status_code)
        print("Some error occured getting data from tmdb api. ID ->", str(tmdbId) )
        return None
    except Exception as e:
      print("GetMovieCover() : Some internal error occured. ", e )
      return None

def GetDetails(movieId):
    


    try:
      tmdbId = int(GetImdbID(movieId))

      req = s.get("https://api.themoviedb.org/3/movie/"  + str(tmdbId) +  "?api_key="  + API_KEY)

      if req.status_code == 200:
        
        tmdbData = json.loads(req.text)
        
        return tmdbData

      else:
        #print(req.status_code)
        print("Some error occured getting data from tmdb api. ID ->", str(tmdbId), req.status_code )
        return None
    except Exception as e:
      print("GetDetails() : Some internal error occured. ", e )
      return None

def GetMostPopularCast(movieId):
    try:
      tmdbId = int(GetImdbID(movieId))

      req = s.get("https://api.themoviedb.org/3/movie/"  + str(tmdbId) +  "/credits?api_key="  + API_KEY)

      if req.status_code == 200:
        
        tmdbData = json.loads(req.text)
        
        most_popular_cast_member =  tmdbData["cast"][0]
        director = None

        for i in range(0, len(tmdbData["cast"])):

            if(float(most_popular_cast_member["popularity"]) < float(tmdbData["cast"][i]["popularity"])):
                most_popular_cast_member =  tmdbData["cast"][i]
            
        for i in range(0, len(tmdbData["crew"])):
            if tmdbData["crew"][i]["job"] == "director":
                director = tmdbData["crew"][i]
        return most_popular_cast_member, director

      else:
        #print(req.status_code)
        print("Some error occured getting data from tmdb api. ID ->", str(tmdbId), req.status_code )
        return None
    except Exception as e:
      print("GetMostPopularCast() : Some internal error occured. ", e )
      return None

movies["processed"] = 0

def Batch1():
  for i in range(0, int(len(movies)/4)):
    print("BATCH 1", i)
    try:
      if(movies.iloc[i]["processed"] == 0 ):
        movie_data = GetDetails(movies.iloc[i]["movieId"].item())
        popular_cast_member, director = GetMostPopularCast(movies.iloc[i]["movieId"].item())
        cover = GetMovieCover(movies.iloc[i]["movieId"].item())

        if(movie_data != None):
            if("release_date" in movie_data):
                movies.loc[i, "release_date"] = int(movie_data["release_date"].split("-")[0])
            if("vote_average" in movie_data):
                movies.loc[i, "vote_average"] = movie_data["vote_average"]
            if("revenue" in movie_data):
                movies.loc[i, "revenue"] = movie_data["revenue"]
        if(popular_cast_member != None):
            if("name" in popular_cast_member):
                movies.loc[i, "popular_cast"] = popular_cast_member["name"]
        if(director != None):
            if("name" in director):
                movies.loc[i, "director"] = director["name"]
        if(cover != None):
            movies.loc[i, "cover"] = cover
        movies.loc[i, "processed"] = 1
        if(i % 100 == 0):
          print("saving")
          movies.to_csv(folder_location + "movies.csv", index=False) 
    except Exception as e:
        print(e)



def Batch2():
  for i in range(int(len(movies)/4), int(len(movies)/2)):
    print("BATCH 2", i)
    try:
      if(movies.iloc[i]["processed"] == 0 ):
        movie_data = GetDetails(movies.iloc[i]["movieId"].item())
        popular_cast_member, director = GetMostPopularCast(movies.iloc[i]["movieId"].item())
        cover = GetMovieCover(movies.iloc[i]["movieId"].item())

        if(movie_data != None):
            if("release_date" in movie_data):
                movies.loc[i, "release_date"] = int(movie_data["release_date"].split("-")[0])
            if("vote_average" in movie_data):
                movies.loc[i, "vote_average"] = movie_data["vote_average"]
            if("revenue" in movie_data):
                movies.loc[i, "revenue"] = movie_data["revenue"]
        if(popular_cast_member != None):
            if("name" in popular_cast_member):
                movies.loc[i, "popular_cast"] = popular_cast_member["name"]
        if(director != None):
            if("name" in director):
                movies.loc[i, "director"] = director["name"]
        if(cover != None):
            movies.loc[i, "cover"] = cover
        movies.loc[i, "processed"] = 1
        if(i % 100 == 0):
          print("saving")
          movies.to_csv(folder_location + "movies.csv", index=False) 
    except Exception as e:
        print(e)
def Batch3():
  for i in range(int(len(movies)/2), len(movies)- int(len(movies)/4)):
    print("BATCH 3", i)
    try:
      if(movies.iloc[i]["processed"] == 0 ):
        movie_data = GetDetails(movies.iloc[i]["movieId"].item())
        popular_cast_member, director = GetMostPopularCast(movies.iloc[i]["movieId"].item())
        cover = GetMovieCover(movies.iloc[i]["movieId"].item())

        if(movie_data != None):
            if("release_date" in movie_data):
                movies.loc[i, "release_date"] = int(movie_data["release_date"].split("-")[0])
            if("vote_average" in movie_data):
                movies.loc[i, "vote_average"] = movie_data["vote_average"]
            if("revenue" in movie_data):
                movies.loc[i, "revenue"] = movie_data["revenue"]
        if(popular_cast_member != None):
            if("name" in popular_cast_member):
                movies.loc[i, "popular_cast"] = popular_cast_member["name"]
        if(director != None):
            if("name" in director):
                movies.loc[i, "director"] = director["name"]
        if(cover != None):
            movies.loc[i, "cover"] = cover
        movies.loc[i, "processed"] = 1
        if(i % 100 == 0):
          print("saving")
          movies.to_csv(folder_location + "movies.csv", index=False) 
    except Exception as e:
        print(e)

def Batch4():
  for i in range( len(movies)- int(len(movies)/4), len(movies)):
    print("BATCH 4", i)
    try:
      if(movies.iloc[i]["processed"] == 0 ):
        movie_data = GetDetails(movies.iloc[i]["movieId"].item())
        popular_cast_member, director = GetMostPopularCast(movies.iloc[i]["movieId"].item())
        cover = GetMovieCover(movies.iloc[i]["movieId"].item())

        if(movie_data != None):
            if("release_date" in movie_data):
                movies.loc[i, "release_date"] = int(movie_data["release_date"].split("-")[0])
            if("vote_average" in movie_data):
                movies.loc[i, "vote_average"] = movie_data["vote_average"]
            if("revenue" in movie_data):
                movies.loc[i, "revenue"] = movie_data["revenue"]
        if(popular_cast_member != None):
            if("name" in popular_cast_member):
                movies.loc[i, "popular_cast"] = popular_cast_member["name"]
        if(director != None):
            if("name" in director):
                movies.loc[i, "director"] = director["name"]
        if(cover != None):
            movies.loc[i, "cover"] = cover
        movies.loc[i, "processed"] = 1
        if(i % 100 == 0):
          print("saving")
          movies.to_csv(folder_location + "movies.csv", index=False) 
    except Exception as e:
        print(e)
import threading

threading.Thread(target=Batch1).start()
threading.Thread(target=Batch2).start()
threading.Thread(target=Batch3).start()
threading.Thread(target=Batch4).start()