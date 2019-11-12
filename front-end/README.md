# GDCR Homepage

## Development

#### Configuration file

Create the `.env` file, add the follow content:

```
BROWSER=none
ESLINT=1
```

#### Dependencies installation & run

```bash
npm i && npm start
```

#### Depoloyment

```bash
npm i && npm run build
```

#### nginx configuration

```nginx
server {
  listen 80;
  listen [::]:80;

  server_name ce7454.pupboss.com;

  location / {
    root              /path/to/demo/dist;
    try_files         $uri /index.html;
  }
}
```
