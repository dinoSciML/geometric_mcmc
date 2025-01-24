class SurrogateModel(object):

    def eval(self, mr):
        
        raise NotImplementedError("")
    
    def jacobian(self, mr):
        raise NotImplementedError("")
    
    def cost(self, ur, mr):
        raise NotImplementedError("")
    
    def misfit_vector(self, ur, mr, weighted=False):
        raise NotImplementedError("")