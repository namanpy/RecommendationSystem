import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

folder_location = "Dataset/"
model_location = "Model/model_rev1.pth"
movies = pd.read_csv( folder_location + "movies_training.csv")
ratings = pd.read_csv( folder_location + "ratings_training.csv", nrows=1001000)

ratings = ratings.dropna()
movies = movies.dropna()

ratings= ratings.reset_index(drop=True)
movies = movies.reset_index(drop=True)





moviesCopy = movies.iloc[:]

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




def MakeList(genreString):
  genres = genreString.split("|")
  binaryGenres = []
  for key in genresHashmap.keys():
    if(key in genres):
      binaryGenres.append(1)
    else:
      binaryGenres.append(0)
  return binaryGenres
def PrepareDataSet():
  if False:
    userHashMap = {}
    lastId = 0
    for i in range(0, len(ratings)):
      if( ratings.loc[i, "userId"].item() in userHashMap.keys()):
        #ratings.loc[i, "userId"] =  userHashMap[ratings.loc[i, "userId"].item()]
        continue
      else:
        lastId += 1
        userHashMap[ratings.loc[i, "userId"].item()] = lastId
        #ratings.loc[i, "userId"] =  lastId

      if(i % 1000 == 0):
        print(i)
    for key in userHashMap.keys():
      print(key, userHashMap[key] )
      ratings.loc[ratings.userId == key, "userId"] = userHashMap[key]
  if False:
    movies["oldMovieId"] = movies["movieId"].copy(deep=True)


    for i in range(0, len(movies)):
      movies.loc[i, "movieId"] = i + 1

    for i in range(0, len(ratings)):

      if(not movies[movies.oldMovieId == ratings.iloc[i]["movieId"]].empty):
        ratings.loc[i, "movieId"] = movies[movies.oldMovieId == ratings.iloc[i]["movieId"].item()]["movieId"].item()
        if(i % 1000 == 0):
          print(i)
      else:
        ratings.loc[i, "movieId"] = np.nan
  maxCombinedGenres = 0.0
  for i in range(0, len(movies)):
    movieGenreHash[movies.loc[i, "movieId"].item()] = (MakeList(movies.loc[i,"genres"]))
  # for key in movieGenreHash.keys():
  #   if(len(movieGenreHash[key]) > maxCombinedGenres ):
  #     maxCombinedGenres = len(movieGenreHash[key])
  # for key in movieGenreHash.keys():
  #   movieGenreHash[key] = torch.FloatTensor( movieGenreHash[key] + [0] * ( maxCombinedGenres - len(movieGenreHash[key]) ) )
  # return maxCombinedGenres




PrepareDataSet()

# movies.to_csv( folder_location + "movies_training.csv", index=False)
# ratings.to_csv( folder_location + "ratings_training.csv", index=False)

print("dataset done")
#print(movieGenreHash)

class RatingDataset(Dataset):
  def __init__(self, ratingData):
    self.data = ratingData
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    try:
      return (
            torch.LongTensor([self.data.loc[idx, "userId"].item()]),
            torch.LongTensor([self.data.loc[idx, "movieId"].item()]),
            torch.Tensor([movieGenreHash[self.data.loc[idx, "movieId"]]]),
            torch.Tensor([[self.data.loc[idx, "vote_average"].item() / 10.0]]),
            torch.Tensor([[2021 - self.data.loc[idx, "release_date"].item()]]),
            torch.Tensor([[self.data.loc[idx, "rating"].item() / 5.0]])
            )
    except Exception as e:
      print("##### Index is  ",  idx)
      print(self.data.loc[idx, "userId"])
    #   torch.Tensor(self.data.loc[idx, "userId"].item()),  
    #   torch.Tensor(self.data.loc[idx, "movieId"].item()),
    #   torch.Tensor(movieGenreHash[self.data.loc[idx, "movieId"]]), 
    #   torch.Tensor([self.data.loc[idx, "vote_average"].item() / 10.0]), 
    #   torch.Tensor([self.data.loc[idx, "release_date"].item() / 2021.0]), 
    #   torch.Tensor([self.data.loc[idx, "rating"].item() / 5.0 ])
    
    # )



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
    print("=>", userId, movieId)
    print( (movieId) )
    print( (userId) )
    print((genre))
    print(vote_average, release_date)

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

    

usersLen = int(ratings.userId.max()) + 1
moviesLen = int(ratings.movieId.max()) + 1

model = WreckEm(moviesLen, usersLen)
try:
  model.load_state_dict(torch.load(model_location))
except FileNotFoundError as e:
  print("No save file found.")

lossFn = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


data = pd.merge(ratings, movies, on="movieId")

data = data.dropna()

data = data.sample(frac=1).reset_index(drop=True)

print(data.vote_average.head())

dataset = RatingDataset(data)

dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

epochs = 10
def Run():
  iters = 1
  lossOverTime  = 0 
  for batch in (dataset):
    userId, movieId, genre, vote_average, release_date, rating = batch
    
    # userId = torch.LongTensor( [userId] )
    # movieId = torch.LongTensor( [movieId] )
    

    optimizer.zero_grad()

    output = model(userId, movieId, genre, vote_average, release_date)
    if(output is not None):
      loss = lossFn(output, rating)

      loss.backward()
      optimizer.step()

      iters += 1
      lossOverTime += loss.item()
      if(iters % 1000 == 0):
        
        print("Loss is ", lossOverTime/iters)
        print(output.item(), rating.item())
      if(iters % 10000 == 0):
        print("Saving..")
        torch.save(model.state_dict(), model_location)


##############################################################################

#                                Testing

##############################################################################

def Test():
  model.eval()
  data = pd.merge(ratings, movies, on="movieId")

  data = data.dropna()

  print(data.head())
  dataset = RatingDataset(data)

  iters = 0

  averageAccuracy = 0
  n = 1
  for i in range(len(ratings) - 1000, len(dataset)):
    userId, movieId, genre, vote_average, release_date, rating = dataset[i]
    


    output = model(userId, movieId, genre, vote_average, release_date)
    accuracy = rating.item() / (rating.item() + abs(rating.item() - output.item()) )
    #accuracy = 1 - abs(accuracy) 
    accuracy *= 100
    print(accuracy, output.item(), rating.item())
    averageAccuracy += accuracy
    n +=  1
  print("Average accuracy is ", averageAccuracy/n ,"%")


def RunBatch():
  runningLoss = 0
  print("Running Batch")
  for i in range(epochs):
    for idx, batch in enumerate(dataloader):
      userId, movieId, genre, vote_average, release_date, rating = batch
      
      optimizer.zero_grad()

      output = model(userId, movieId, genre, vote_average, release_date)

      if(output is not None):
            loss = lossFn(output, rating)

            loss.backward()
            optimizer.step()

            iters += 1
            lossOverTime += loss.item()
            if(iters % 1000 == 0):
              
              print("Loss is ", lossOverTime/iters)
              print(output.item(), rating.item())
            if(iters % 10000 == 0):
              print("Saving..")
              torch.save(model.state_dict(), model_location)

#Run()
Test()