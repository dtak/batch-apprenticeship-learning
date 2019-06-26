class ActWrapper(object):
    def __init__(self, act):
        self.act = act

    def choose_action(self, s):
        return self.act(s[None])[0]

