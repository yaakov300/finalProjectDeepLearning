from api.models import Networks

app_networks = Networks()
app_networks.load_existing_networks()

def update_app_networks(new_model):
    global app_networks
    app_networks.networks = new_model

