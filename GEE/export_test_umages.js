// https://code.earthengine.google.com/bd4f007eae5379388ee6c65e9004d4ea

var sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR"),
    modis_landcover = ee.ImageCollection("MODIS/006/MCD12Q1"),
    s2_rgb_params = {"opacity":1,"bands":["B4","B3","B2"],"max":2800,"gamma":1},
    s2_fci_params = {"opacity":1,"bands":["B8","B4","B3"],"max":5500,"gamma":1},
    landcover_params = {"opacity":1,"bands":["LC_Type2"],"max":14,"gamma":1},
    Munich = 
    /* shown: false */
    ee.Geometry.Polygon(
        [[[11.187300272361282, 48.371390591601504],
          [11.187300272361282, 47.80816968479942],
          [11.978315897361282, 47.80816968479942],
          [11.978315897361282, 48.371390591601504]]], null, false),
    Rome = 
    /* shown: false */
    ee.Geometry.Polygon(
        [[[12.253784717277933, 42.10769338579563],
          [12.253784717277933, 41.74088048700778],
          [12.718643726066995, 41.74088048700778],
          [12.718643726066995, 42.10769338579563]]], null, false),
    StPetersburg = 
    /* shown: false */
    ee.Geometry.Polygon(
        [[[29.624200370680633, 60.12804588053008],
          [29.624200370680633, 59.688761064696415],
          [30.659661796461883, 59.688761064696415],
          [30.659661796461883, 60.12804588053008]]], null, false);

var ROI = StPetersburg;  // one of: Munich, Rome, StPetersburg

var s2_image = sentinel2.filterBounds(ROI)
    .filterDate('2017-01-01', '2017-08-31')
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 1))
    .mosaic()
    .clip(ROI);

var landcover = modis_landcover.filter(ee.Filter.eq('system:index', '2017_01_01'))
    .mosaic()
    .clip(ROI)
    .select('LC_Type2');

Map.centerObject(ROI, 10);
Map.addLayer(landcover, landcover_params, 'Landcover', false);
Map.addLayer(s2_image, s2_rgb_params, 'Sentinel-2 RGB', true);
Map.addLayer(s2_image, s2_fci_params, 'Sentinel-2 FCI', false);

// split the Sentinel-2 image into parts, so that the exported
// is not too massive for out little GDrives

var s2_part1 = s2_image.select('B2', 'B3', 'B4');
var s2_part2 = s2_image.select('B8', 'B5', 'B6');
var s2_part3 = s2_image.select('B7', 'B8A', 'B11');
var s2_part4 = s2_image.select('B12');

Export.image.toDrive({
  'image': s2_part4,
  'description': 'stpetersburg_s2_se12ms_b2b3b4',
  'scale': 10,
  'crs': 'EPSG:32636'  // for StPetersburg: EPSG:32636, for Rome and Munich: EPSG:32633
});
