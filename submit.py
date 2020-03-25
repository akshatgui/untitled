import client_moodle as cli, pickle
fname = 'topvec.pkl'
f = open(fname, 'rb')
vector = pickle.load(f)
res1 = cli.get_errors('jcikTU98ZdeaUH5uBHsOPXzXAzAhBVdwtDDj7SqoF98mqbjZLw', vector)
print(res1)
res2 = cli.submit('jcikTU98ZdeaUH5uBHsOPXzXAzAhBVdwtDDj7SqoF98mqbjZLw', vector)
print(res2)