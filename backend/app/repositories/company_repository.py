from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.company import Company


class CompanyRepository:
    def get_by_ticker(self, db: Session, ticker: str) -> Company | None:
        return db.scalar(select(Company).where(Company.ticker == ticker))

    def create(self, db: Session, company: Company) -> Company:
        db.add(company)
        return company
