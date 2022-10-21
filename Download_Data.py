import os
import numpy as np
import rasterio as rio
import io
import requests
import ee
from os.path import exists
service_account = 'geo-test@geotest-317218.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, 'geotest-privkey.json')
ee.Initialize(credentials)
os.chdir('/home/hamtech/Downloads/Data')


# library for unzip tiff image that downloaded from google earth engine
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import requests


class Processor:
    def __init__(self, start, end, north, south, East, West, pixles, count):
        self.start = start
        self.end = end

        self.north = north
        self.south = south
        self.East = East
        self.West = West
        self.pixles = pixles
        self.polygon = []
        self.count = count
    ######################################################################
    def geojson(self):
        vertical_coords = np.linspace(self.south, self.north, self.pixles + 1)
        horizontal_coords = np.linspace(self.West, self.East, self.pixles + 1)

        North_coords = vertical_coords[1:]
        South_coords = vertical_coords[:-1]
        West_coords = horizontal_coords[:-1]
        East_coords = horizontal_coords[1:]

        coords_list = []
        for i in range(pixles):
            for j in range(pixles):
                coords_list.append([
                    [West_coords[j], South_coords[i]],
                    [West_coords[j], North_coords[i]],
                    [East_coords[j], North_coords[i]],
                    [East_coords[j], South_coords[i]],
                    [West_coords[j], South_coords[i]],
                ])

        coords_list = coords_list[self.count:]
        for coordinates in coords_list:
            geoJSON = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                coordinates
                            ]
                        }
                    }
                ]
            }
            coords = geoJSON['features'][0]['geometry']['coordinates']
            self.polygon.append(ee.Geometry.Polygon(coords))

    ######################################################################
    def get_images_sentinel2(self, coordinates):
        s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'TCI_R', 'TCI_G', 'TCI_B']
        image = (ee.ImageCollection("COPERNICUS/S2_SR")
                 .filterDate(self.start, self.end)
                 .select(s2_bands)
                 .filterBounds(coordinates)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                 .mean()
                 .clip(coordinates))
        return image

    ######################################################################
    def get_images_Land_use(self, coordinates):
        image = ee.ImageCollection("ESA/WorldCover/v100").first().clip(coordinates)
        return image

    ######################################################################
    def turn_image_to_raster(self, image, title, coordinate, folder):
        # download image from google earth engine
        url = image.getDownloadURL(
            params={'name': title, 'scale': 10, 'region': coordinate,
                    'crs': 'EPSG:4326', 'filePerBand': False,'format': 'GEO_TIFF'})

        response = requests.get(url)
        with open(folder + title + '.tif', 'wb') as fd:
            fd.write(response.content)

    ##################################################################################################
    def main(self):
        O = 0
        if not os.path.isdir('lable'):
            os.mkdir('lable')
        if not os.path.isdir('sentinel1'):
            os.mkdir('sentinel1')
        if not os.path.isdir('sentinel2'):
            os.mkdir('sentinel2')

        self.geojson()
        k = self.count
        for polygon in self.polygon:
            lable = self.get_images_Land_use(coordinates=polygon)
            self.turn_image_to_raster(image=lable, title='lable' + str(k), coordinate=polygon, folder='./lable/')


            sentinel2 = self.get_images_sentinel2(coordinates=polygon)
            self.turn_image_to_raster(image=sentinel2, title='sentinel2_' + str(k), coordinate=polygon,
                                      folder='./sentinel2/')

            k+=1
            print(f'Loop {O} out of {len(self.polygon)}')
            O += 1


directory = ['pol5']
start_time = '2020-01-01'
end_time = '2021-01-01'
north = [37.458]
south = [36.870]
East = [49.328]
West = [48.74]
pixles_list = [12]   # 0, 2, 3, 6, 8, needs rotate,
count = 0
for i in range(len(north)):
    pixles = pixles_list[i]
    print(f'Polygon {i} out of {len(north)} has been started')
    os.chdir(directory[i])
    p1 = Processor(start_time, end_time, north=north[i], south=south[i], East=East[i], West=West[i], pixles=pixles, count=count)
    p1.main()
    count = 0


