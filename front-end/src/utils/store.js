import cookie from 'js-cookie'

const PREFIX = 'gdcr'

export function set(key, value, exp) {
  if (typeof document !== 'undefined') {
    const days = (exp && exp > 0) ? exp / (1000 * 3600 * 24) : null
    cookie.set(`${PREFIX}.${key}`, value, { expires: days })
  }
}

export function get(key) {
  if (typeof document !== 'undefined') {
    return cookie.get(`${PREFIX}.${key}`)
  }
}

export function remove(key) {
  if (typeof document !== 'undefined') {
    return cookie.remove(`${PREFIX}.${key}`)
  }
}
