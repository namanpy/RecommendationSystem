import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

genresHashmap = {
  "Action" : 1,
  "Adventure" : 2,
  "Animation" : 3,
  "Children" : 4,
  "Comedy" : 5,
  "Crime" : 6,
  "Documentary" : 7,
  "Drama" : 8,
  "Fantasy" : 9,
  "Film-Noir" : 10,
  "Horror" : 11,
  "Musical" : 12,
  "Mystery" : 13,
  "Romance" : 14,
  "Sci-Fi"  : 15,
  "Thriller" : 16,
  "War" : 17,
  "Western" : 18,
  "IMAX" : 19,
  "(no genres listed)" : 20
}

movieGenreHash = {

}

class WreckEm(nn.Module):
  def __init__(self, moviesLen, usersLen):
    super(WreckEm, self).__init__()

    print("=>", usersLen, moviesLen)
    self.movie = nn.Embedding(moviesLen, 20)
    self.user = nn.Embedding(usersLen,  20)
    
    self.genreFc = nn.Linear(len(genresHashmap.keys()) , 16)
    
    self.fc1 = nn.Linear(57, 128)
    self.fc2 = nn.Linear(128 ,32)
    self.fc3 = nn.Linear(32, 1)

    self.drop = torch.nn.Dropout(p=0.08)
  def forward(self, userId, movieId, genre, vote_average, release_date):
    # print("=>", userId, movieId)
    # print( (movieId) )
    # print( (userId) )
    # print((genre))
    # print(vote_average, release_date)

    try:
      x = torch.cat(  [ self.movie(movieId), F.relu(self.genreFc(genre)), self.user(userId), vote_average], 1) # 
      # x = torch.cat([x ,], 1)
      x = F.relu(self.fc1(x))
      x = self.drop(x)
      x = F.relu(self.fc2(x))
      x = self.drop(x)
      x = F.sigmoid(self.fc3(x))
      return x
    except Exception as e:
      print(e)
      print( userId, movieId, genre, vote_average, release_date)
      return None
def ParseGenre(genreString):
  genres = genreString.split("|")
  binaryGenres = []
  for key in genresHashmap.keys():
    if(key in genres):
      binaryGenres.append(1)
    else:
      binaryGenres.append(0)
  return binaryGenres