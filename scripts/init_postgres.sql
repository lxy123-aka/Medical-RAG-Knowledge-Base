-- =============================================================================
-- PostgreSQL 建表 SQL - 医典 RAG 多维知识库系统
-- =============================================================================

-- 文档元数据表
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(128) UNIQUE NOT NULL,       -- 文档唯一标识
    title VARCHAR(512),                         -- 文档标题
    source_type VARCHAR(64) NOT NULL,           -- 来源类型: textbook/record/pharmacopoeia
    department VARCHAR(128),                    -- 所属科室
    permission_level INTEGER DEFAULT 0,         -- 权限等级
    page_count INTEGER,                         -- 页数
    file_path VARCHAR(1024),                    -- 文件路径
    created_at TIMESTAMP DEFAULT NOW(),         -- 创建时间
    updated_at TIMESTAMP DEFAULT NOW()          -- 更新时间
);

-- 文档分片表
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(128) REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,               -- 分片序号
    content TEXT NOT NULL,                      -- 分片内容
    token_count INTEGER,                        -- token 数量
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(doc_id, chunk_index)
);

-- 病历信息表
CREATE TABLE IF NOT EXISTS medical_records (
    id SERIAL PRIMARY KEY,
    record_id VARCHAR(128) UNIQUE NOT NULL,     -- 病历编号
    patient_id VARCHAR(128) NOT NULL,           -- 患者编号
    doctor_name VARCHAR(128),                   -- 主治医生
    department VARCHAR(128),                    -- 科室
    admission_date DATE,                        -- 入院日期
    discharge_date DATE,                        -- 出院日期
    chief_complaint TEXT,                       -- 主诉
    diagnosis TEXT,                             -- 诊断
    treatment_plan TEXT,                        -- 治疗方案
    permission_level INTEGER DEFAULT 0,         -- 权限等级
    created_at TIMESTAMP DEFAULT NOW()
);

-- 药物信息表
CREATE TABLE IF NOT EXISTS drugs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(256) NOT NULL,                 -- 药物名称
    generic_name VARCHAR(256),                  -- 通用名
    dosage_form VARCHAR(128),                   -- 剂型
    manufacturer VARCHAR(256),                  -- 生产厂家
    approval_number VARCHAR(128),               -- 批准文号
    contraindications TEXT,                     -- 禁忌症
    side_effects TEXT,                          -- 副作用
    dosage_instructions TEXT,                   -- 用法用量
    permission_level INTEGER DEFAULT 0,         -- 权限等级
    created_at TIMESTAMP DEFAULT NOW()
);

-- 疾病信息表
CREATE TABLE IF NOT EXISTS diseases (
    id SERIAL PRIMARY KEY,
    name VARCHAR(256) NOT NULL,                 -- 疾病名称
    icd_code VARCHAR(32),                       -- ICD 编码
    category VARCHAR(128),                      -- 分类
    description TEXT,                           -- 描述
    prevalence VARCHAR(128),                    -- 患病率
    common_symptoms TEXT,                       -- 常见症状
    permission_level INTEGER DEFAULT 0,         -- 权限等级
    created_at TIMESTAMP DEFAULT NOW()
);

-- =============================================================================
-- 索引
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_documents_source_type ON documents(source_type);
CREATE INDEX IF NOT EXISTS idx_documents_department ON documents(department);
CREATE INDEX IF NOT EXISTS idx_documents_permission ON documents(permission_level);
CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_id ON document_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_medical_records_patient ON medical_records(patient_id);
CREATE INDEX IF NOT EXISTS idx_medical_records_department ON medical_records(department);
CREATE INDEX IF NOT EXISTS idx_medical_records_permission ON medical_records(permission_level);
CREATE INDEX IF NOT EXISTS idx_diseases_name ON diseases(name);
CREATE INDEX IF NOT EXISTS idx_diseases_icd ON diseases(icd_code);
CREATE INDEX IF NOT EXISTS idx_drugs_name ON drugs(name);

-- =============================================================================
-- 示例数据
-- =============================================================================
INSERT INTO diseases (name, icd_code, category, description, common_symptoms)
VALUES
    ('高血压', 'I10', '心血管', '以动脉血压持续升高为特征的慢性病', '头痛,头晕,心悸'),
    ('2型糖尿病', 'E11', '内分泌', '以慢性高血糖为特征的代谢性疾病', '多饮,多尿,体重下降'),
    ('冠心病', 'I25.1', '心血管', '冠状动脉粥样硬化导致的心肌缺血', '胸闷,心悸,气短'),
    ('肺炎', 'J18', '呼吸', '肺部感染性炎症', '咳嗽,发热,胸痛'),
    ('胃炎', 'K29', '消化', '胃黏膜炎症', '腹痛,恶心,腹胀')
ON CONFLICT DO NOTHING;

INSERT INTO drugs (name, generic_name, dosage_form, contraindications, side_effects)
VALUES
    ('硝苯地平', 'Nifedipine', '缓释片', '严重主动脉瓣狭窄', '面部潮红,头痛,水肿'),
    ('二甲双胍', 'Metformin', '片剂', '严重肾功能不全', '恶心,腹泻,腹痛'),
    ('阿司匹林', 'Aspirin', '肠溶片', '活动性消化道出血', '胃肠道出血,过敏'),
    ('阿莫西林', 'Amoxicillin', '胶囊', '青霉素过敏', '过敏反应,腹泻'),
    ('奥美拉唑', 'Omeprazole', '肠溶胶囊', '对质子泵抑制剂过敏', '头痛,腹泻,恶心')
ON CONFLICT DO NOTHING;
