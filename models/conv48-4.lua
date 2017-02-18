nn = require 'nn'

-- hybrated

local features = nn.Sequential()
features:add(nn.SpatialConvolution(3,27,7,7))      
features:add(nn.Tanh())
features:add(nn.SpatialMaxPooling(2,2,2,2))        

features:add(nn.SpatialConvolution(27,128,6,6))    
features:add(nn.Tanh())
features:add(nn.SpatialMaxPooling(2,2,2,2))        

features:add(nn.SpatialConvolution(128, 320, 4, 4))
features:add(nn.Tanh())
features:add(nn.SpatialMaxPooling(2,2,2,2))


features:add(nn.View(1280)) -- 320 * 2 * 2
features:add(nn.Dropout(0.5))

features:add(nn.Linear(1280, 320))
features:add(nn.Tanh())
features:add(nn.Dropout(0.5))

features:add(nn.Linear(320, 43))

return features