CREATE TABLE `products`
(
    `product_id`           int NOT NULL AUTO_INCREMENT COMMENT '제품 ID',
    `prod_name`            varchar(255) COMMENT '제품 이름',
    `description`          text COMMENT '제품에 대한 설명',
    `price`                varchar(20) COMMENT '제품의 가격',
    `url`                  varchar(255) COMMENT '상품 URL',
    `create_date`          varchar(255) COMMENT '등록일',
    `avg_rating`           int COMMENT '평균 평점',
    `brand_name`           varchar(255) COMMENT '판매사',
    `positive_reviews_cnt` int COMMENT '긍정 리뷰 수',
    `negative_reviews_cnt` int COMMENT '부정 리뷰 수',
    'product_summary'     varchar(255) COMMENT '제품 요약(모델링)',
    PRIMARY KEY (`product_id`)
) COMMENT = '제품 정보 테이블';



CREATE TABLE `reviews`
(
    `review_id`       int NOT NULL AUTO_INCREMENT COMMENT '리뷰 ID',
    `prod_id`         int COMMENT '제품 ID',
    `prod_name`       varchar(255) COMMENT '제품 이름',
    `rating`          varchar(10) COMMENT '리뷰 평점',
    `title`           varchar(255) COMMENT '리뷰 제목',
    `context`         text COMMENT '리뷰 내용',
    `answer`          varchar(255) COMMENT '리뷰 총평',
    `review_url`      varchar(255) COMMENT '리뷰 URL',
    `helped_cnt`      int COMMENT '도움이 된 수',
    `create_date`     varchar(255) COMMENT '작성일',
    `top100_yn`       varchar(1) COMMENT '리뷰작성자 TOP 100 여부',
    `sentiment`       varchar(1) COMMENT '긍부정판단(모델링)',
    PRIMARY KEY (`review_id`)
) COMMENT = '리뷰 정보 테이블';



delete
from salmon.products;

delete
from reviews;


select p.prod_name,
       r.rating,
       r.title,
       r.context,
       r.answer,
       r.review_url,
       r.helped_cnt,
       r.create_date,
       r.top100_yn,
       r.sentiment,
       r.keywords,
       r.search_caterory
from products as p
         left join reviews as r on p.product_id = r.prod_id
where p.prod_name like '%밀키트%';
