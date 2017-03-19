nn = require 'nn'

-- all Tanh()

local model = nn.Sequential()

model:add(nn.SpatialConvolution(3,27,8,8))       
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(2,2,2,2))         

model:add(nn.SpatialConvolution(27,128,5,5))     
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(2,2,2,2))         

model:add(nn.View(128*8*8))
model:add(nn.Dropout(0.5))

model:add(nn.Linear(128*8*8, 320))
model:add(nn.Tanh())

model:add(nn.Dropout(0.5))

model:add(nn.Linear(320, 43))

return model