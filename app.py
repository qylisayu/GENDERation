import os
import shutil
import atexit
import dash

app = dash.Dash(__name__, use_pages=True, external_stylesheets=['assets/styles.css'])
app.layout = dash.page_container

def delete_local_files():
    if os.path.exists('upload'):
        shutil.rmtree('upload')

atexit.register(delete_local_files)

if __name__ == '__main__':
	app.run_server(debug=True)
