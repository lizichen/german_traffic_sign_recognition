nn = require 'nn'

local features = nn.Sequential()
features:add(nn.SpatialConvolution(3,100,7,7))      -- 1
features:add(nn.ReLU(true))
features:add(nn.SpatialMaxPooling(2,2,2,2):ceil())         -- 2

features:add(nn.SpatialConvolution(100,150,4,4))    -- 3
features:add(nn.ReLU(true))
features:add(nn.SpatialMaxPooling(2,2,2,2):ceil())         -- 4

features:add(nn.SpatialConvolution(150,250,4,4))    -- 5
features:add(nn.ReLU(true))
features:add(nn.SpatialMaxPooling(2,2,2,2):ceil())         -- 6

features:add(nn.View(2250))  -- 250*3*3
features:add(nn.Dropout(0.5))

features:add(nn.Linear(2250, 300))				    -- 7
features:add(nn.ReLU(true))
features:add(nn.Dropout(0.5))

features:add(nn.Linear(300, 43))						   -- 8

return features