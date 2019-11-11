import Header from './Header'
import Footer from './Footer'
import styles from './index.css';

function BasicLayout(props) {
  return (
    <div className={styles.normal}>
      <Header />
      <div className={styles.content}>{props.children}</div>
      <Footer />
    </div>
  );
}

export default BasicLayout;
