import React from 'react'
import { Link } from 'dva/router'

import styles from './Header.css'

const Header = () => (
  <div className={styles.normal}>
    <Link className={styles.logo} to="/">
      <img height="40px" alt="logo" src="/image/logo-300x300.png" />
    </Link>
  </div>
)

export default Header
