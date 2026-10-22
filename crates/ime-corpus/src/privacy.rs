//! The personal-data gate: sentences that carry someone's contact or identity.
//!
//! Scraped forums and web pages hold phone numbers, resident identity numbers,
//! email addresses and street addresses, and the anonymous boards (北大树洞 and
//! its kin) hold them about other people. None of it is anything an input method
//! should learn to complete, and a target cut from such a sentence can carry the
//! Han half of an address into training, so the whole sentence goes -- as a
//! target and as context for its neighbours -- before any target is cut.
//!
//! The rules match the shapes the data uses rather than validating them: a
//! mistyped identity number is as personal as a correct one. They run on
//! normalised text, so full-width digits have already been folded to ASCII.

use regex::Regex;

/// What kind of personal data a dropped sentence held.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PersonalKind {
    /// A mainland mobile or landline number.
    Phone,
    /// A resident identity number, eighteen or fifteen digits.
    IdNumber,
    /// An email address.
    Email,
    /// A street address with a house, building or room number.
    Address,
}

/// How many sentences each rule dropped, for the census.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct PersonalCounts {
    /// Sentences dropped for a phone number.
    pub phone: usize,
    /// Sentences dropped for an identity number.
    pub id_number: usize,
    /// Sentences dropped for an email address.
    pub email: usize,
    /// Sentences dropped for a street address.
    pub address: usize,
}

impl PersonalCounts {
    /// Count one dropped sentence under its kind.
    pub fn record(&mut self, kind: PersonalKind) {
        match kind {
            PersonalKind::Phone => self.phone += 1,
            PersonalKind::IdNumber => self.id_number += 1,
            PersonalKind::Email => self.email += 1,
            PersonalKind::Address => self.address += 1,
        }
    }

    /// Fold another tally into this one.
    pub fn merge(&mut self, other: Self) {
        self.phone += other.phone;
        self.id_number += other.id_number;
        self.email += other.email;
        self.address += other.address;
    }

    /// Every sentence dropped, whatever it held.
    #[must_use]
    pub fn total(&self) -> usize {
        self.phone + self.id_number + self.email + self.address
    }
}

/// Digits as an address writes them: Arabic, or Chinese numerals.
const NUMBER: &str = "[0-9零〇一二三四五六七八九十百两]+";

/// The compiled rules, built once per run and shared across threads.
#[derive(Clone, Debug)]
pub struct PersonalData {
    rules: [(PersonalKind, Regex); 4],
}

impl PersonalData {
    /// Compile the rules.
    ///
    /// # Panics
    ///
    /// Never at run time: the patterns are constants, and the tests compile them.
    #[must_use]
    pub fn new() -> Self {
        let compile = |pattern: &str| Regex::new(pattern).expect("the pattern is a constant");
        // Every number rule is bounded by a non-digit (or an end) on both sides,
        // so a long run of digits -- an order number, a hash -- is not read as a
        // phone number that happens to sit inside it.
        let phone = compile(concat!(
            r"(?:^|[^0-9])(?:",
            // Mobile: 1[3-9]x xxxx xxxx, with an optional +86 and separators.
            r"(?:\+?86[- ]?)?1[3-9][0-9](?:[- ]?[0-9]{4}){2}",
            r"|",
            // Landline: an area code starting with 0, then seven or eight digits.
            r"(?:\(0[0-9]{2,3}\)|0[0-9]{2,3})[- ]?[2-9][0-9]{6,7}",
            r")(?:[^0-9]|$)",
        ));
        let id_number = compile(concat!(
            r"(?:^|[^0-9])(?:",
            // Eighteen digits: region, birth date, sequence, checksum digit or X.
            r"[1-9][0-9]{5}(?:18|19|20)[0-9]{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12][0-9]|3[01])[0-9]{3}[0-9Xx]",
            r"|",
            // Fifteen digits: the pre-1999 form with a two-digit year.
            r"[1-9][0-9]{5}[0-9]{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12][0-9]|3[01])[0-9]{3}",
            r")(?:[^0-9A-Za-z]|$)",
        ));
        let email = compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}");
        // A street with a house or lane number (建国路88号, 人民街道三号,
        // 长乐路123弄), or a building, unit and room (3栋2单元502, 5号楼1203室).
        // A street name alone is a public place and stays; `号线` is a metro
        // line, and a bare `道` is too often 知道 or 难道 before a date.
        let address = compile(&format!(
            "(?:路|街|街道|大道|巷|胡同){NUMBER}(?:号|弄)(?:[^线]|$)\
             |[0-9]+(?:栋|幢|号楼|座)[0-9]+单元(?:[0-9]{{3,4}})?\
             |[0-9]+(?:栋|幢|号楼|单元)[0-9]{{3,4}}(?:室|房)?\
             |[^0-9][0-9]{{3,4}}室"
        ));
        Self {
            rules: [
                (PersonalKind::Email, email),
                (PersonalKind::IdNumber, id_number),
                (PersonalKind::Phone, phone),
                (PersonalKind::Address, address),
            ],
        }
    }

    /// The first kind of personal data `sentence` holds, if any.
    #[must_use]
    pub fn find(&self, sentence: &str) -> Option<PersonalKind> {
        self.rules
            .iter()
            .find(|(_, rule)| rule.is_match(sentence))
            .map(|(kind, _)| *kind)
    }
}

impl Default for PersonalData {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind(sentence: &str) -> Option<PersonalKind> {
        PersonalData::new().find(sentence)
    }

    #[test]
    fn a_mobile_number_is_found_in_every_way_it_is_written() {
        for sentence in [
            "有事打我电话13812345678谢谢。",
            "联系方式：138-1234-5678。",
            "微信同号 138 1234 5678",
            "请拨+86 13912345678咨询。",
            "号码是8613712345678哦",
        ] {
            assert_eq!(kind(sentence), Some(PersonalKind::Phone), "{sentence}");
        }
    }

    #[test]
    fn a_landline_number_is_found_with_or_without_its_separator() {
        for sentence in [
            "座机010-62751234转分机。",
            "电话(021)55667788。",
            "拨打075588886666",
        ] {
            assert_eq!(kind(sentence), Some(PersonalKind::Phone), "{sentence}");
        }
    }

    #[test]
    fn an_identity_number_is_found_in_both_lengths() {
        assert_eq!(
            kind("身份证号110105199003071234，请核对。"),
            Some(PersonalKind::IdNumber)
        );
        assert_eq!(
            kind("证件11010519900307123X。"),
            Some(PersonalKind::IdNumber)
        );
        assert_eq!(
            kind("旧证号110105900307123。"),
            Some(PersonalKind::IdNumber)
        );
    }

    #[test]
    fn an_email_address_is_found() {
        assert_eq!(
            kind("简历发到 zhang.san+jobs@mail.example.cn 就行。"),
            Some(PersonalKind::Email)
        );
    }

    #[test]
    fn a_street_address_with_a_number_is_found() {
        for sentence in [
            "我家住在建国路88号。",
            "地址：人民街道三号院。",
            "搬到了幸福小区3栋2单元502。",
            "在5号楼1203室等你。",
            "快递放到1203室门口。",
            "南锣鼓巷十二号的那家店。",
            "住在长乐路123弄。",
        ] {
            assert_eq!(kind(sentence), Some(PersonalKind::Address), "{sentence}");
        }
    }

    #[test]
    fn ordinary_numbers_and_public_places_are_left_alone() {
        for sentence in [
            "今天天气不错。",
            "南京路是一条有名的步行街。",
            "坐地铁4号线到西单。",
            "这套房子三室两厅。",
            "2023年GDP增长了5.2%。",
            "订单号20231215000123456789已发货。",
            "人口约13亿。",
            "他考了650分。",
            "比分是3:1。",
            "@张翼高飞 你可以玩这个了",
            "我知道3号要考试。",
            "难道3号不行吗？",
        ] {
            assert_eq!(kind(sentence), None, "{sentence}");
        }
    }

    #[test]
    fn counts_fold_together_by_kind() {
        let mut counts = PersonalCounts::default();
        counts.record(PersonalKind::Phone);
        counts.record(PersonalKind::Address);
        let mut other = PersonalCounts::default();
        other.record(PersonalKind::Phone);
        counts.merge(other);
        assert_eq!(counts.phone, 2);
        assert_eq!(counts.address, 1);
        assert_eq!(counts.total(), 3);
    }
}
