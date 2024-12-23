from EnvWithObstacles import Continuous2DEnvWithRectObstaclesBox

# env=Continuous2DEnvWithRectObstaclesBox(start=[1,1],
#                goal=[18,15],
#                 layout="simple")
env=Continuous2DEnvWithRectObstaclesBox(start=[1,1],
               goal=[14,17.5],
                layout="moderate")
# env=Continuous2DEnvWithRectObstaclesBox(start=[2.5,17.5],
#                goal=[16,1],
#                 layout="complex")
env.render()