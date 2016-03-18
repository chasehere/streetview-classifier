import sys
import urllib
import pandas as pd

# TODO: provide some updates, this takes awhile, some timing too

def main(argv):
  
  print "Loading data from input file %s..." % argv[0]
  data = pd.read_csv(argv[0])

  print "Downloading images from google..."
  for index, row in data.iterrows():
    #if index == 0:
      outfile = "images/" + str(row["parcel_number"]) + "_" + str(row["street_view_rating"]) + ".jpg"
      location = row["address"] 
      url = "http://maps.googleapis.com/maps/api/streetview?size=128x128&location=" + location + "&key=AIzaSyBnjld-ZFxZe0npFvybCWpb3d1WS16iIww"
      urllib.urlretrieve(url,outfile)

  

if __name__ == "__main__":
  # first arg is input file, this should be a list of addresses
  # second arg is output file
  main(sys.argv[1:])
