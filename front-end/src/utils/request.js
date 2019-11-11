import 'whatwg-fetch'

function parse(response) {
  const { status } = response
  if (response.ok) {
    if (status === 204) {
      return {}
    }
    return response.json()
  }
  return response.json().then((data) => {
    const error = new Error(data.message)
    error.status = status
    error.data = data
    error.code = data.code
    error.detail = data.detail
    throw error
  })
}

export default async function request(url, opts = {}) {

  const headers = {
    ...opts.headers,
    'Content-Type': 'application/x-www-form-urlencoded',
    'Accept': 'application/json',
  }

  const options = {
    ...opts,
    headers,
  }

  try {
    const result = await fetch(url, options)
      .then(parse)
      .then(data => ({ data }))
      .catch((err) => {
        if (err.status && err.status < 500) {
          return { err: { code: `${err.status}`, ...err.data } }
        }
        throw err
      })
    return result
  } catch (err) {
    return { err: new Error('API_SERVER_INTERNAL_ERROR') }
  }
}
