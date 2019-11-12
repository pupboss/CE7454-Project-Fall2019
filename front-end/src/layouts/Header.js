import React from 'react'
import { router } from 'dva'
import { Typography } from 'antd'
import styles from './Header.css'

const { Title } = Typography
const { Link } = router;

const Header = () => (
  <div className={styles.normal}>
    <Link className={styles.logo} to="/">
      <img height="40px" alt="logo" src="/image/logo-300x300.png" />
    </Link>
    <div className={styles.title}>
      <Title level={2}>Simultaneous Prediction of Box-ofﬁce and Movie Rating</Title>
    </div>
  </div>
)

export default Header
