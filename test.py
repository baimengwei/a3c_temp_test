import gym
env = gym.make('LunarLander-v2')
s = env.reset()
for i in range(100):
    env.render()
    a = env.action_space.sample()
    ns, r, d, _ = env.step(a)
    print(s, a, r, ns, d, _)
    input('input:')
    if d:
        break
    s = ns
