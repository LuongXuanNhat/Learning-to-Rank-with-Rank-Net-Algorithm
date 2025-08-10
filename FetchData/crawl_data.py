import time
import json
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import os

# ✅ CHỈ ĐỊNH rõ chrome driver và chrome binary
chrome_driver_path = r"D:\BTMONHOC\SĐH\KhaiThacThongTin\FetchData\chromedriver.exe"
chrome_binary_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cadao_crawler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CaDaoTucNguCrawler:
    def __init__(self, url="https://loigiaihay.com/tuyen-tap-2000-cau-tuc-ngu-viet-nam-e31016.html"):
        self.url = url
        self.driver = None
        self.cadao_list = []
        self.all_crawled_texts = set()  # Tập hợp để tránh trùng lặp
        self.current_id = 1  # ID counter
        
    def setup_driver(self):
        """Khởi tạo Chrome driver với các options tối ưu"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Chạy ngầm, không hiển thị browser
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(30)
            logger.info("Chrome driver đã được khởi tạo thành công")
        except Exception as e:
            logger.error(f"Lỗi khởi tạo Chrome driver: {e}")
            raise
    
    def load_page(self):
        """Load trang web"""
        try:
            logger.info(f"Đang tải trang: {self.url}")
            self.driver.get(self.url)
            
            # Đợi trang load
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            logger.info("Trang đã được tải thành công")
            
        except TimeoutException:
            logger.error(f"Timeout khi tải trang {self.url}")
            raise
        except Exception as e:
            logger.error(f"Lỗi khi tải trang: {e}")
            raise
    
    def expand_all_buttons(self):
        """Mở rộng tất cả các button 'Xem thêm'"""
        logger.info("Bắt đầu tìm và mở rộng tất cả button 'Xem thêm'...")
        
        total_clicked = 0
        max_attempts = 20  # Giới hạn số lần thử
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Tìm tất cả các button "Xem thêm" có class "item-more-wiki"
                more_buttons = self.driver.find_elements(By.CSS_SELECTOR, "span.item-more-wiki")
                
                if not more_buttons:
                    logger.info("Không còn button 'Xem thêm' nào")
                    break
                
                logger.info(f"Lần thử {attempt + 1}: Tìm thấy {len(more_buttons)} button 'Xem thêm'")
                
                # Click vào từng button
                clicked_this_round = 0
                for i, button in enumerate(more_buttons):
                    try:
                        # Kiểm tra xem button có hiển thị không
                        if not button.is_displayed():
                            continue
                        
                        # Cuộn đến button
                        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
                        time.sleep(0.5)
                        
                        # Click bằng JavaScript
                        self.driver.execute_script("arguments[0].click();", button)
                        
                        clicked_this_round += 1
                        total_clicked += 1
                        
                        logger.info(f"Đã click button {clicked_this_round}")
                        
                        # Đợi nội dung load
                        time.sleep(1)
                        
                    except Exception as e:
                        logger.warning(f"Không thể click button {i+1}: {e}")
                        continue
                
                if clicked_this_round == 0:
                    logger.info("Không click được button nào trong lần này - có thể đã hết")
                    break
                    
                logger.info(f"Đã click {clicked_this_round} button trong lần thử này")
                
                # Đợi trang load trước khi tìm button tiếp theo
                time.sleep(2)
                attempt += 1
                
            except Exception as e:
                logger.error(f"Lỗi khi mở rộng button: {e}")
                break
        
        logger.info(f"Hoàn thành mở rộng button. Tổng cộng đã click {total_clicked} button")
        return total_clicked > 0
    
    def extract_cadao_content(self):
        """Trích xuất tất cả nội dung ca dao từ trang"""
        try:
            logger.info("Bắt đầu trích xuất nội dung ca dao...")
            
            # Đợi một chút để đảm bảo tất cả nội dung đã load
            time.sleep(3)
            
            # Tìm tất cả các thẻ wiki-article có chứa link (không phải button)
            content_articles = self.driver.find_elements(By.CSS_SELECTOR, ".wiki-article")
            
            logger.info(f"Tìm thấy {len(content_articles)} thẻ wiki-article tổng cộng")
            
            extracted_count = 0
            for i, article in enumerate(content_articles):
                try:
                    # Bỏ qua các thẻ chỉ chứa button "Xem thêm"
                    if article.find_elements(By.CSS_SELECTOR, "span.item-more-wiki"):
                        continue
                    
                    # Tìm thẻ <a> chứa nội dung ca dao
                    link_elements = article.find_elements(By.TAG_NAME, "a")
                    
                    for link in link_elements:
                        cadao_text = link.text.strip()
                        
                        # Kiểm tra text hợp lệ
                        if cadao_text and len(cadao_text) > 5:
                            # Kiểm tra trùng lặp
                            if cadao_text not in self.all_crawled_texts:
                                self.all_crawled_texts.add(cadao_text)
                                
                                # Thêm vào danh sách với cấu trúc id và value
                                self.cadao_list.append({
                                    "id": self.current_id,
                                    "value": cadao_text
                                })
                                
                                self.current_id += 1
                                extracted_count += 1
                                
                                if extracted_count % 100 == 0:
                                    logger.info(f"Đã trích xuất {extracted_count} ca dao...")
                            else:
                                logger.debug(f"Bỏ qua ca dao trùng lặp: {cadao_text[:50]}...")
                
                except Exception as e:
                    logger.warning(f"Lỗi khi xử lý article {i}: {e}")
                    continue
            
            logger.info(f"Hoàn thành trích xuất: {extracted_count} ca dao/tục ngữ UNIQUE")
            return extracted_count
            
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất nội dung: {e}")
            return 0
    
    def save_to_json(self):
        """Lưu tất cả ca dao vào file JSON"""
        try:
            if not self.cadao_list:
                logger.warning("Không có dữ liệu để lưu")
                return False
            
            # Tạo thư mục output nếu chưa có
            os.makedirs("output", exist_ok=True)
            filepath = os.path.join("output", "cadao_tucngu_complete.json")
            
            with open(filepath, 'w', encoding='utf-8') as jsonfile:
                json.dump(self.cadao_list, jsonfile, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Đã lưu {len(self.cadao_list)} ca dao vào file {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu file JSON: {e}")
            return False
    
    def print_statistics(self):
        """In thống kê dữ liệu đã crawl"""
        logger.info("=== THỐNG KÊ DỮ LIỆU ===")
        logger.info(f"Tổng số ca dao/tục ngữ: {len(self.cadao_list)}")
        logger.info(f"ID cao nhất: {self.current_id - 1}")
        
        if len(self.cadao_list) > 0:
            logger.info("=== VÍ DỤ 5 CA DAO ĐẦU TIÊN ===")
            for i in range(min(5, len(self.cadao_list))):
                cadao = self.cadao_list[i]
                logger.info(f"ID {cadao['id']}: {cadao['value']}")
    
    def cleanup(self):
        """Đóng driver và dọn dẹp"""
        if self.driver:
            self.driver.quit()
            logger.info("Đã đóng Chrome driver")
    
    def run(self):
        """Chạy toàn bộ quá trình crawling"""
        try:
            logger.info("=== BẮT ĐẦU CRAWLING CA DAO TỤC NGỮ ===")
            
            # Bước 1: Khởi tạo driver
            self.setup_driver()
            
            # Bước 2: Load trang
            self.load_page()
            
            # Bước 3: Mở rộng tất cả button "Xem thêm"
            expanded = self.expand_all_buttons()
            
            # Bước 4: Trích xuất nội dung
            extracted_count = self.extract_cadao_content()
            
            # Bước 5: Lưu vào file JSON
            if extracted_count > 0:
                success = self.save_to_json()
                if success:
                    self.print_statistics()
                    logger.info("=== HOÀN THÀNH THÀNH CÔNG ===")
                else:
                    logger.error("=== LỖI KHI LƯU FILE ===")
            else:
                logger.warning("=== KHÔNG TÌM THẤY DỮ LIỆU ===")
            
        except KeyboardInterrupt:
            logger.info("Người dùng dừng chương trình")
        except Exception as e:
            logger.error(f"Lỗi không mong muốn: {e}")
        finally:
            # Dọn dẹp
            self.cleanup()

# Chạy crawler
if __name__ == "__main__":
    crawler = CaDaoTucNguCrawler()
    crawler.run()